package ro.lab.ai.ga;

import java.util.List;

/**
 * Obiectele JSON schimbate cu front-end-ul pentru algoritmul genetic.
 * Vectorii de gene circulă ca șiruri de 64 de caractere '0'/'1'.
 */
public final class GaApi {

    private GaApi() {
    }

    public record ResetRequest(Integer populationSize, Double mutationRate, String crossover, Boolean elitism) {}

    public record StepRequest(Integer generations, Double mutationRate, String crossover, Boolean elitism) {}

    /** Fie {@code target} (șir de 64 biți), fie {@code preset} (numele unui model). */
    public record TargetRequest(String target, String preset) {}

    public record IndividualDto(String genes, int fitness, boolean elite) {}

    public record CrossDto(String parentA, String parentB, String child, String origin,
                          int fitnessA, int fitnessB, int fitnessChild, int[] cuts, int[] mutated) {}

    /** Istoricul (eventual eșantionat, ca să nu crească la nesfârșit); {@code generations} = câte puncte reale există. */
    public record History(int generations, List<Integer> best, List<Double> average, List<Integer> worst) {}

    public record Snapshot(
            String target,
            int length,
            int generation,
            int best,
            double average,
            int worst,
            long crossCount,
            boolean solved,
            int populationSize,
            double mutationRate,
            String crossover,
            boolean elitism,
            List<IndividualDto> population,
            int bestChildIndex,
            CrossDto lastCross,
            double[] genePool,
            History history) {}
}
