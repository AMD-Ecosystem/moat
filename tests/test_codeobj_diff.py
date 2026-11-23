# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import os
import pathlib
import struct
import tempfile
import unittest
from unittest import mock

import codeobj_diff  # utils/ is put on sys.path by tests/__init__.py

MAGIC = codeobj_diff.BUNDLE_MAGIC
GFX = b"hipv4-amdgcn-amd-amdhsa--gfx942"
HOST = b"host-x86_64-unknown-linux-gnu"


def bundle(*entries, data_at=256):
    """One clang offload bundle: entries are (triple, payload)."""
    head = MAGIC + struct.pack("<Q", len(entries))
    body, cursor = b"", data_at
    for triple, payload in entries:
        head += struct.pack("<QQQ", cursor, len(payload), len(triple)) + triple
        body += payload
        cursor += len(payload)
    assert len(head) <= data_at
    return head + b"\0" * (data_at - len(head)) + body


def elf(sections, machine=62):
    """A minimal little-endian ELF64 holding the named sections."""
    names = b"\0"
    offs = {}
    for n in list(sections) + [".shstrtab"]:
        offs[n] = len(names)
        names += n.encode() + b"\0"
    blobs = dict(sections)
    blobs[".shstrtab"] = names
    data, where = b"", {}
    for n, payload in blobs.items():
        where[n] = (64 + len(data), len(payload))
        data += payload
    shoff = 64 + len(data)
    heads = struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    for n in blobs:
        off, size = where[n]
        heads += struct.pack("<IIQQQQIIQQ", offs[n], 1, 0, 0, off, size, 0, 0, 1, 0)
    ident = b"\x7fELF" + bytes([2, 1, 1, 0]) + b"\0" * 8
    header = ident + struct.pack("<HHIQQQIHHHHHH", 3, machine, 1, 0, 0, shoff, 0,
                                 64, 0, 0, 64, len(blobs) + 1, len(blobs))
    return header + data + heads


class Sections(unittest.TestCase):
    def test_finds_a_section_by_name(self):
        image = elf({".text": b"abcd", ".hip_fatbin": b"x" * 10})
        machine, secs = codeobj_diff._elf_sections(image)
        self.assertEqual(machine, 62)
        off, size = secs[".hip_fatbin"]
        self.assertEqual(image[off:off + size], b"x" * 10)

    def test_not_elf(self):
        self.assertIsNone(codeobj_diff._elf_sections(b"MZ" + b"\0" * 100))


class Bundles(unittest.TestCase):
    def slices(self, section):
        image = elf({".hip_fatbin": section})
        _m, secs = codeobj_diff._elf_sections(image)
        found = codeobj_diff._bundle_slices(image, *secs[".hip_fatbin"])
        return image, found

    def test_one_bundle_host_entry_is_skipped(self):
        image, found = self.slices(bundle((HOST, b""), (GFX, b"CODEOBJ1")))
        self.assertEqual([(t, image[o:o + n]) for t, o, n in found],
                         [(GFX.decode(), b"CODEOBJ1")])

    def test_one_bundle_per_translation_unit_with_padding_between(self):
        section = (bundle((GFX, b"first")) + b"\0" * 37
                   + bundle((HOST, b""), (GFX, b"second")) + b"\0" * 5)
        image, found = self.slices(section)
        self.assertEqual([image[o:o + n] for _t, o, n in found], [b"first", b"second"])

    def test_magic_inside_a_code_object_is_not_a_header(self):
        image, found = self.slices(bundle((GFX, b"xx" + MAGIC + b"yy")))
        self.assertEqual(len(found), 1)

    def test_unaccounted_bytes_void_the_answer(self):
        _image, found = self.slices(bundle((GFX, b"first")) + b"CCOB\x01\x02\x03")
        self.assertIsNone(found)

    def test_compressed_only_section_voids_the_answer(self):
        _image, found = self.slices(b"CCOB" + b"\x07" * 40)
        self.assertIsNone(found)

    def test_entry_running_past_the_section_voids_the_answer(self):
        head = MAGIC + struct.pack("<Q", 1) + struct.pack("<QQQ", 64, 9999, len(GFX)) + GFX
        _image, found = self.slices(head + b"\0" * 64)
        self.assertIsNone(found)

    def test_empty_section_has_no_device_code(self):
        _image, found = self.slices(b"\0" * 16)
        self.assertEqual(found, [])


class DeviceIsa(unittest.TestCase):
    def isa(self, image):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bin")
            with open(path, "wb") as f:
                f.write(image)
            fake = mock.Mock(returncode=0, stdout="   0: aa bb \tnop\n")
            with mock.patch.object(codeobj_diff, "_tool", lambda name: "objdump"), \
                    mock.patch.object(codeobj_diff, "_run", lambda args: fake):
                return codeobj_diff._device_isa(path, d)

    def test_host_only_binary_is_an_answer_not_a_failure(self):
        self.assertEqual(self.isa(elf({".text": b"abcd"})), "")

    def test_device_code_is_disassembled_per_target(self):
        out = self.isa(elf({".hip_fatbin": bundle((GFX, b"CODE"))}))
        self.assertIn(GFX.decode(), out)
        self.assertIn("nop", out)

    def test_a_bare_code_object_is_its_own_device_code(self):
        out = self.isa(elf({".text": b"abcd"}, machine=codeobj_diff.EM_AMDGPU))
        self.assertIn("nop", out)

    def test_unreadable_bundle_is_a_failure(self):
        self.assertIsNone(self.isa(elf({".hip_fatbin": b"CCOB" + b"\x07" * 40})))


class Gather(unittest.TestCase):
    def test_cmake_probe_binaries_are_not_project_output(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "CMakeFiles", "3.28", "CompilerIdCXX"))
            for rel in ("libfoo.so", os.path.join("CMakeFiles", "3.28",
                                                  "CompilerIdCXX", "a.out")):
                with open(os.path.join(d, rel), "wb") as f:
                    f.write(b"\x7fELF" + b"\0" * 60)
            self.assertEqual(sorted(codeobj_diff._gather(d)), ["libfoo.so"])


class WindowsBinaries(unittest.TestCase):
    def test_pe_binaries_are_gathered_and_answered_indeterminate(self):
        with tempfile.TemporaryDirectory() as tmp:
            for side in ("old", "new"):
                d = pathlib.Path(tmp, side)
                d.mkdir()
                (d / "core.dll").write_bytes(b"MZ" + b"\0" * 62)
            self.assertEqual(sorted(codeobj_diff._gather(str(pathlib.Path(tmp, "old")))),
                             ["core.dll"])
            with mock.patch.object(codeobj_diff, "_tool", lambda name: name):
                verdict, details = codeobj_diff.compare(
                    str(pathlib.Path(tmp, "old")), str(pathlib.Path(tmp, "new")))
        self.assertEqual(verdict, "indeterminate")
        self.assertIn("PE/COFF", details[0])


class Tools(unittest.TestCase):
    def test_missing_tools_are_indeterminate_and_say_why(self):
        with mock.patch.object(codeobj_diff, "_tool", lambda name: None):
            verdict, why = codeobj_diff.compare_binary("a", "b")
        self.assertEqual(verdict, "indeterminate")
        self.assertIn("ROCM_PATH", why)


if __name__ == "__main__":
    unittest.main()
