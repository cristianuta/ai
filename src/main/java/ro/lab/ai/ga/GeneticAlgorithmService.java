package ro.lab.ai.ga;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.springframework.stereotype.Service;
import org.springframework.web.context.annotation.SessionScope;

import ro.lab.ai.ga.GaApi.CrossDto;
import ro.lab.ai.ga.GaApi.History;
import ro.lab.ai.ga.GaApi.IndividualDto;
import ro.lab.ai.ga.GaApi.Snapshot;
import ro.lab.ai.ga.GeneticAlgorithm.Crossover;
import ro.lab.ai.ga.GeneticAlgorithm.Individual;

/** Starea algoritmului genetic pentru un utilizator (o sesiune HTTP). */
@Service
@SessionScope
public class GeneticAlgorithmService {

    /** Câte puncte de istoric trimitem maxim către grafic. */
    static final int MAX_HISTORY_POINTS = 600;

    private final Random rnd = new Random();
    private GeneticAlgorithm ga;

    public GeneticAlgorithmService() {
        ga = new GeneticAlgorithm(GaPresets.get("heart", rnd), 40, 0.015, Crossover.ONE_POINT, true, rnd);
    }

    public synchronized void reset(int populationSize, double mutationRate, Crossover crossover, boolean elitism) {
        ga = new GeneticAlgorithm(ga.target(), populationSize, mutationRate, crossover, elitism, rnd);
    }

    /** Rulează până la {@code generations} generații (se oprește mai devreme dacă ținta e atinsă). */
    public synchronized void step(int generations, Double mutationRate, Crossover crossover, Boolean elitism) {
        if (mutationRate != null) ga.setMutationRate(mutationRate);
        if (crossover != null) ga.setCrossover(crossover);
        if (elitism != null) ga.setElitism(elitism);
        if (ga.isSolved()) ga.reset(); // pornim o evoluție nouă după ce ținta a fost atinsă
        for (int g = 0; g < generations && !ga.isSolved(); g++) ga.step();
    }

    public synchronized void setTarget(byte[] target) {
        ga.setTarget(target);
    }

    public synchronized void setPreset(String preset) {
        ga.setTarget(GaPresets.get(preset, rnd));
    }

    public synchronized Snapshot snapshot() {
        List<Individual> pop = ga.population();
        List<IndividualDto> dtos = new ArrayList<>(pop.size());
        Individual lastCross = ga.lastCross();
        int bestChildIndex = -1;
        for (int k = 0; k < pop.size(); k++) {
            Individual ind = pop.get(k);
            if (ind == lastCross) bestChildIndex = k;
            dtos.add(new IndividualDto(bits(ind.genes()), ind.fitness(), ind.elite()));
        }
        CrossDto cross = null;
        if (lastCross != null) {
            GeneticAlgorithm.CrossInfo info = lastCross.info();
            cross = new CrossDto(bits(info.parentA()), bits(info.parentB()), bits(lastCross.genes()), bits(info.origin()),
                    info.fitnessA(), info.fitnessB(), lastCross.fitness(), info.cuts(), info.mutated());
        }
        History history = new History(ga.bestHistory().size(),
                sample(ga.bestHistory()), sample(ga.averageHistory()), sample(ga.worstHistory()));
        return new Snapshot(
                bits(ga.target()), ga.target().length, ga.generation(),
                ga.best(), ga.average(), ga.worst(), ga.crossCount(), ga.isSolved(),
                ga.populationSize(), ga.mutationRate(), ga.crossover().code(), ga.elitism(),
                dtos, bestChildIndex, cross, ga.genePool(), history);
    }

    static String bits(byte[] genes) {
        StringBuilder sb = new StringBuilder(genes.length);
        for (byte g : genes) sb.append(g == 1 ? '1' : '0');
        return sb.toString();
    }

    /** Eșantionează uniform o listă lungă, păstrând primul și ultimul punct. */
    static <T> List<T> sample(List<T> list) {
        int n = list.size();
        if (n <= MAX_HISTORY_POINTS) return List.copyOf(list);
        List<T> out = new ArrayList<>(MAX_HISTORY_POINTS);
        for (int k = 0; k < MAX_HISTORY_POINTS; k++) {
            out.add(list.get((int) Math.round((double) k * (n - 1) / (MAX_HISTORY_POINTS - 1))));
        }
        return out;
    }
}
