package ro.lab.ai.ga;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import ro.lab.ai.ga.GaApi.ResetRequest;
import ro.lab.ai.ga.GaApi.Snapshot;
import ro.lab.ai.ga.GaApi.StepRequest;
import ro.lab.ai.ga.GaApi.TargetRequest;
import ro.lab.ai.ga.GeneticAlgorithm.Crossover;

/** API pentru algoritmul genetic. Fiecare operație întoarce starea completă ({@link Snapshot}). */
@RestController
@RequestMapping("/api/ga")
public class GeneticAlgorithmController {

    private final GeneticAlgorithmService ga;

    public GeneticAlgorithmController(GeneticAlgorithmService ga) {
        this.ga = ga;
    }

    @GetMapping("/state")
    public Snapshot state() {
        return ga.snapshot();
    }

    @PostMapping("/reset")
    public Snapshot reset(@RequestBody ResetRequest req) {
        int pop = req.populationSize() == null ? 40 : req.populationSize();
        if (pop < 2 || pop > 500) throw new IllegalArgumentException("Populația trebuie să fie între 2 și 500");
        ga.reset(pop, mutationRate(req.mutationRate(), 0.015),
                req.crossover() == null ? Crossover.ONE_POINT : Crossover.fromCode(req.crossover()),
                req.elitism() == null || req.elitism());
        return ga.snapshot();
    }

    @PostMapping("/step")
    public Snapshot step(@RequestBody StepRequest req) {
        int generations = req.generations() == null ? 1 : Math.max(1, Math.min(500, req.generations()));
        ga.step(generations,
                req.mutationRate() == null ? null : mutationRate(req.mutationRate(), 0),
                req.crossover() == null ? null : Crossover.fromCode(req.crossover()),
                req.elitism());
        return ga.snapshot();
    }

    @PostMapping("/target")
    public Snapshot target(@RequestBody TargetRequest req) {
        if (req.preset() != null) {
            ga.setPreset(req.preset());
        } else if (req.target() != null && req.target().matches("[01]{64}")) {
            byte[] t = new byte[64];
            for (int i = 0; i < 64; i++) t[i] = (byte) (req.target().charAt(i) - '0');
            ga.setTarget(t);
        } else {
            throw new IllegalArgumentException("Ținta trebuie să fie un șir de 64 de caractere 0/1 sau un model");
        }
        return ga.snapshot();
    }

    private static double mutationRate(Double v, double fallback) {
        if (v == null) return fallback;
        if (!(v >= 0 && v <= 1)) throw new IllegalArgumentException("Rata de mutație trebuie să fie între 0 și 1");
        return v;
    }
}
