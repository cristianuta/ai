package ro.lab.ai.ga;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

/**
 * Algoritm genetic pe vectori binari.
 * <ul>
 *   <li>fitness: numărul de gene identice cu ținta</li>
 *   <li>selecție: turneu (k = 3)</li>
 *   <li>încrucișare: un punct / două puncte / uniformă</li>
 *   <li>mutație: fiecare genă se inversează cu probabilitatea {@code mutationRate}</li>
 *   <li>elitism opțional: cel mai bun individ trece neschimbat în generația următoare</li>
 * </ul>
 */
public class GeneticAlgorithm {

    public enum Crossover {
        ONE_POINT("one"), TWO_POINT("two"), UNIFORM("uniform");

        private final String code;

        Crossover(String code) {
            this.code = code;
        }

        public String code() {
            return code;
        }

        public static Crossover fromCode(String code) {
            for (Crossover c : values()) if (c.code.equals(code)) return c;
            throw new IllegalArgumentException("Tip de încrucișare necunoscut: " + code);
        }
    }

    /** De unde provine un copil: genele părinților, ce bucată a luat de la fiecare și ce s-a mutat. */
    public record CrossInfo(byte[] parentA, byte[] parentB, int fitnessA, int fitnessB,
                            byte[] origin, int[] cuts, int[] mutated) {}

    /** Un individ: vectorul de gene + fitness. {@code info == null} pentru elită / populația inițială. */
    public static final class Individual {
        private final byte[] genes;
        private final CrossInfo info;
        private final boolean elite;
        private int fitness;

        Individual(byte[] genes, CrossInfo info, boolean elite) {
            this.genes = genes;
            this.info = info;
            this.elite = elite;
        }

        public byte[] genes() { return genes; }
        public CrossInfo info() { return info; }
        public boolean elite() { return elite; }
        public int fitness() { return fitness; }
    }

    record CrossResult(byte[] child, byte[] origin, int[] cuts) {}

    private final Random rnd;
    private final int tournamentSize = 3;
    private byte[] target;
    private int populationSize;
    private double mutationRate;
    private Crossover crossover;
    private boolean elitism;

    private List<Individual> population;
    private Individual lastCross;
    private int generation;
    private long crossCount;
    private int best;
    private double average;
    private int worst;
    private final List<Integer> bestHistory = new ArrayList<>();
    private final List<Double> averageHistory = new ArrayList<>();
    private final List<Integer> worstHistory = new ArrayList<>();

    public GeneticAlgorithm(byte[] target, int populationSize, double mutationRate,
                            Crossover crossover, boolean elitism, Random rnd) {
        this.rnd = rnd;
        this.target = target.clone();
        this.populationSize = populationSize;
        this.mutationRate = mutationRate;
        this.crossover = crossover;
        this.elitism = elitism;
        reset();
    }

    /** Populație nouă, complet aleatoare. */
    public final void reset() {
        generation = 0;
        crossCount = 0;
        lastCross = null;
        bestHistory.clear();
        averageHistory.clear();
        worstHistory.clear();
        population = new ArrayList<>(populationSize);
        for (int p = 0; p < populationSize; p++) {
            byte[] genes = new byte[target.length];
            for (int i = 0; i < genes.length; i++) genes[i] = (byte) (rnd.nextBoolean() ? 1 : 0);
            population.add(new Individual(genes, null, false));
        }
        evaluate(true);
    }

    public int fitness(byte[] genes) {
        int c = 0;
        for (int i = 0; i < genes.length; i++) if (genes[i] == target[i]) c++;
        return c;
    }

    private void evaluate(boolean record) {
        long sum = 0;
        for (Individual ind : population) {
            ind.fitness = fitness(ind.genes);
            sum += ind.fitness;
        }
        population.sort(Comparator.comparingInt(Individual::fitness).reversed());
        best = population.getFirst().fitness;
        worst = population.getLast().fitness;
        average = (double) sum / population.size();
        if (record) {
            bestHistory.add(best);
            averageHistory.add(average);
            worstHistory.add(worst);
        }
    }

    private Individual tournament() {
        Individual winner = null;
        for (int k = 0; k < tournamentSize; k++) {
            Individual c = population.get(rnd.nextInt(population.size()));
            if (winner == null || c.fitness > winner.fitness) winner = c;
        }
        return winner;
    }

    /** Combină doi vectori. {@code origin[i] = 0} dacă gena i vine de la A, 1 dacă vine de la B. */
    CrossResult cross(byte[] a, byte[] b) {
        int n = a.length;
        byte[] origin = new byte[n];
        int[] cuts;
        switch (crossover) {
            case ONE_POINT -> {
                int c = 1 + rnd.nextInt(n - 1);
                cuts = new int[]{c};
                for (int i = c; i < n; i++) origin[i] = 1;
            }
            case TWO_POINT -> {
                int c1 = 1 + rnd.nextInt(n - 1);
                int c2 = 1 + rnd.nextInt(n - 1);
                if (c1 > c2) { int t = c1; c1 = c2; c2 = t; }
                if (c1 == c2) {
                    if (c2 < n - 1) c2++;
                    else c1--;
                }
                cuts = new int[]{c1, c2};
                for (int i = c1; i < c2; i++) origin[i] = 1;
            }
            default -> {
                cuts = new int[0];
                for (int i = 0; i < n; i++) origin[i] = (byte) (rnd.nextBoolean() ? 1 : 0);
            }
        }
        byte[] child = new byte[n];
        for (int i = 0; i < n; i++) child[i] = origin[i] == 1 ? b[i] : a[i];
        return new CrossResult(child, origin, cuts);
    }

    /** O generație: selecție → încrucișare → mutație → evaluare. */
    public void step() {
        List<Individual> next = new ArrayList<>(populationSize);
        if (elitism) next.add(new Individual(population.getFirst().genes.clone(), null, true));
        while (next.size() < populationSize) {
            Individual pa = tournament();
            Individual pb = tournament();
            for (int t = 0; t < 3 && pb == pa; t++) pb = tournament();
            CrossResult r = cross(pa.genes, pb.genes);
            List<Integer> mutated = new ArrayList<>();
            for (int i = 0; i < r.child().length; i++) {
                if (rnd.nextDouble() < mutationRate) {
                    r.child()[i] ^= 1;
                    mutated.add(i);
                }
            }
            crossCount++;
            CrossInfo info = new CrossInfo(pa.genes, pb.genes, pa.fitness, pb.fitness, r.origin(), r.cuts(),
                    mutated.stream().mapToInt(Integer::intValue).toArray());
            next.add(new Individual(r.child(), info, false));
        }
        population = next;
        generation++;
        evaluate(true);
        // cel mai bun copil nou (nu elita copiată) — pentru vizualizarea încrucișării
        for (Individual ind : population) {
            if (ind.info != null) {
                lastCross = ind;
                break;
            }
        }
    }

    public boolean isSolved() {
        return best == target.length;
    }

    /** Pentru fiecare poziție: fracțiunea din populație care are gena corectă. */
    public double[] genePool() {
        double[] pool = new double[target.length];
        for (Individual ind : population) {
            for (int i = 0; i < pool.length; i++) if (ind.genes[i] == target[i]) pool[i]++;
        }
        for (int i = 0; i < pool.length; i++) pool[i] /= population.size();
        return pool;
    }

    // ------------------------------------------------------------------ setări

    public void setTarget(byte[] target) {
        if (target.length != this.target.length) throw new IllegalArgumentException("Ținta are altă lungime");
        this.target = target.clone();
        evaluate(false);
    }

    public void setMutationRate(double mutationRate) { this.mutationRate = mutationRate; }
    public void setCrossover(Crossover crossover) { this.crossover = crossover; }
    public void setElitism(boolean elitism) { this.elitism = elitism; }

    // ------------------------------------------------------------------ citire

    public byte[] target() { return target.clone(); }
    public List<Individual> population() { return Collections.unmodifiableList(population); }
    public Individual lastCross() { return lastCross; }
    public int generation() { return generation; }
    public long crossCount() { return crossCount; }
    public int best() { return best; }
    public double average() { return average; }
    public int worst() { return worst; }
    public int populationSize() { return populationSize; }
    public double mutationRate() { return mutationRate; }
    public Crossover crossover() { return crossover; }
    public boolean elitism() { return elitism; }
    public List<Integer> bestHistory() { return bestHistory; }
    public List<Double> averageHistory() { return averageHistory; }
    public List<Integer> worstHistory() { return worstHistory; }
}
