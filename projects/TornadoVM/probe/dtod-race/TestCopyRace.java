package uk.ac.manchester.tornado.unittests.pointers;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.unittests.common.TornadoTestBase;

// Review-only probe: warm both graphs so no JIT compile separates the copy from the next launch.
public class TestCopyRace extends TornadoTestBase {

    public static void zero(uk.ac.manchester.tornado.api.types.arrays.FloatArray in) {
        for (@uk.ac.manchester.tornado.api.annotations.Parallel int i = 0; i < in.getSize(); i++) {
            in.set(i, -5f);
        }
    }

    public static void reverseRead(uk.ac.manchester.tornado.api.types.arrays.FloatArray in, uk.ac.manchester.tornado.api.types.arrays.FloatArray out) {
        for (@uk.ac.manchester.tornado.api.annotations.Parallel int i = 0; i < out.getSize(); i++) {
            out.set(i, in.get(in.getSize() - 1 - i) + 0.1f);
        }
    }

    @Test
    public void testWarmLargeMapping() throws TornadoExecutionPlanException {
        final int rows = 16384;
        final int columns = 16384;
        final int firstRow = 4096;
        final int mappedRows = 8192;
        FloatArray matrix = new FloatArray(rows * columns);
        TaskGraph g0 = new TaskGraph("r0").task("f0", TestCopyRace::zero, matrix).task("r0", TestCopyDevicePointers::iterativeUpdate2D, matrix, rows, columns).transferToHost(DataTransferMode.UNDER_DEMAND, matrix);
        FloatArray block = new FloatArray(mappedRows * columns);
        FloatArray out = new FloatArray(1024);
        TaskGraph g1 = new TaskGraph("r1").transferToDevice(DataTransferMode.UNDER_DEMAND, block).task("r1", TestCopyRace::reverseRead, block, out).task("z1", TestCopyRace::zero, block).transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        int bad = 0;
        String first = "";
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g0.snapshot(), g1.snapshot())) {
            plan.withGraph(0).execute();
            plan.withGraph(1).execute();
            for (int it = 0; it < 20; it++) {
                plan.withGraph(0).execute();
                plan.mapOnDeviceMemoryRegion(block, matrix, (long) firstRow * columns, 0, 1);
                plan.withGraph(1).execute();
                int[] probes = { 0, 1023 };
                for (int i : probes) {
                    float want = firstRow + (block.getSize() - 1 - i) / columns + 0.1f;
                    if (Math.abs(out.get(i) - want) > 0.01f) {
                        if (bad == 0) {
                            first = "it " + it + " element " + i + " want " + want + " got " + out.get(i);
                        }
                        bad++;
                    }
                }
            }
        }
        System.out.println("RACEPROBE bad=" + bad + " " + first);
        assertEquals(0, bad, first);
    }
}
