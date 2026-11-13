package ro.lab.ai.ga;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.Test;

import ro.lab.ai.ga.GeneticAlgorithm.Crossover;

class GeneticAlgorithmTest {

    private static byte[] ones(int n) {
        byte[] b = new byte[n];
        java.util.Arrays.fill(b, (byte) 1);
        return b;
    }

    @Test
    void onePointCrossoverTakesPrefixFromAAndSuffixFromB() {
        GeneticAlgorithm ga = new GeneticAlgorithm(new byte[64], 10, 0, Crossover.ONE_POINT, true, new Random(3));
        byte[] a = new byte[64];
        byte[] b = ones(64);
        GeneticAlgorithm.CrossResult r = ga.cross(a, b);
        int cut = r.cuts()[0];
        for (int i = 0; i < 64; i++) {
            assertEquals(i < cut ? 0 : 1, r.child()[i]);
            assertEquals(i < cut ? 0 : 1, r.origin()[i]);
        }
    }

    @Test
    void twoPointCrossoverTakesMiddleFromB() {
        GeneticAlgorithm ga = new GeneticAlgorithm(new byte[64], 10, 0, Crossover.TWO_POINT, true, new Random(5));
        for (int t = 0; t < 200; t++) {
            GeneticAlgorithm.CrossResult r = ga.cross(new byte[64], ones(64));
            int c1 = r.cuts()[0], c2 = r.cuts()[1];
            assertTrue(c1 < c2);
            for (int i = 0; i < 64; i++) assertEquals(i >= c1 && i < c2 ? 1 : 0, r.child()[i]);
        }
    }

    @Test
    void solvesRandomTargetsWithEveryCrossover() {
        Random rnd = new Random(11);
        for (Crossover c : Crossover.values()) {
            byte[] target = GaPresets.get("random", rnd);
            GeneticAlgorithm ga = new GeneticAlgorithm(target, 40, 0.015, c, true, rnd);
            while (!ga.isSolved() && ga.generation() < 1000) ga.step();
            assertTrue(ga.isSolved(), c + " nu a găsit ținta");
            assertArrayEquals(target, ga.population().getFirst().genes());
        }
    }

    @Test
    void elitismNeverLosesTheBestIndividual() {
        GeneticAlgorithm ga = new GeneticAlgorithm(GaPresets.get("heart", new Random(1)), 20, 0.1,
                Crossover.UNIFORM, true, new Random(2));
        int previous = ga.best();
        for (int g = 0; g < 100; g++) {
            ga.step();
            assertTrue(ga.best() >= previous);
            previous = ga.best();
        }
    }

    @Test
    void historySamplingKeepsEnds() {
        List<Integer> list = java.util.stream.IntStream.range(0, 5000).boxed().toList();
        List<Integer> s = GeneticAlgorithmService.sample(list);
        assertEquals(GeneticAlgorithmService.MAX_HISTORY_POINTS, s.size());
        assertEquals(0, s.getFirst());
        assertEquals(4999, s.getLast());
    }
}
