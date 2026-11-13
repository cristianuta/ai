package ro.lab.ai.nn;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import ro.lab.ai.nn.NnApi.ActivationsResponse;
import ro.lab.ai.nn.NnApi.DataRequest;
import ro.lab.ai.nn.NnApi.InputRequest;
import ro.lab.ai.nn.NnApi.ResetRequest;
import ro.lab.ai.nn.NnApi.Sample;
import ro.lab.ai.nn.NnApi.SampleRequest;
import ro.lab.ai.nn.NnApi.Snapshot;
import ro.lab.ai.nn.NnApi.TrainRequest;

/**
 * API pentru rețeaua neurală. Majoritatea operațiilor primesc și desenul curent ({@code input})
 * și întorc un {@link Snapshot} complet, ca interfața să poată redesena totul dintr-un singur răspuns.
 */
@RestController
@RequestMapping("/api/nn")
public class NeuralNetworkController {

    private static final int MAX_HIDDEN_LAYERS = 4;
    private static final int MAX_LAYER_SIZE = 64;

    private final NeuralNetworkService nn;

    public NeuralNetworkController(NeuralNetworkService nn) {
        this.nn = nn;
    }

    @PostMapping("/state")
    public Snapshot state(@RequestBody(required = false) InputRequest req) {
        return nn.snapshot(input(req == null ? null : req.input()));
    }

    @PostMapping("/predict")
    public ActivationsResponse predict(@RequestBody InputRequest req) {
        return new ActivationsResponse(nn.activations(input(req.input())));
    }

    @PostMapping("/train")
    public Snapshot train(@RequestBody TrainRequest req) {
        int batches = req.batches() == null ? 1 : clamp(req.batches(), 1, 200);
        nn.train(batches, learningRate(req.learningRate()));
        return nn.snapshot(input(req.input()));
    }

    @PostMapping("/epoch")
    public Snapshot epoch(@RequestBody TrainRequest req) {
        nn.trainEpoch(learningRate(req.learningRate()));
        return nn.snapshot(input(req.input()));
    }

    @PostMapping("/reset")
    public Snapshot reset(@RequestBody ResetRequest req) {
        int[] hidden = req.hidden() == null ? new int[]{16, 16} : req.hidden();
        if (hidden.length < 1 || hidden.length > MAX_HIDDEN_LAYERS) {
            throw new IllegalArgumentException("Între 1 și " + MAX_HIDDEN_LAYERS + " straturi ascunse");
        }
        for (int size : hidden) {
            if (size < 1 || size > MAX_LAYER_SIZE) {
                throw new IllegalArgumentException("Un strat ascuns are între 1 și " + MAX_LAYER_SIZE + " neuroni");
            }
        }
        nn.reset(hidden);
        return nn.snapshot(input(req.input()));
    }

    @PostMapping("/data")
    public Snapshot regenerate(@RequestBody DataRequest req) {
        double noise = req.noise() == null ? 0.15 : req.noise();
        if (noise < 0 || noise > 0.5) throw new IllegalArgumentException("Zgomotul trebuie să fie între 0 și 0.5");
        nn.regenerateData(noise);
        return nn.snapshot(input(req.input()));
    }

    @PostMapping("/samples")
    public Snapshot addSample(@RequestBody SampleRequest req) {
        double[] pixels = input(req.pixels());
        if (req.label() == null || req.label() < 0 || req.label() > 9) {
            throw new IllegalArgumentException("Eticheta trebuie să fie o cifră 0–9");
        }
        boolean blank = true;
        for (double v : pixels) if (v > 0) { blank = false; break; }
        if (blank) throw new IllegalArgumentException("Desenul este gol");
        nn.addSample(pixels, req.label());
        return nn.snapshot(pixels);
    }

    @GetMapping("/random-sample")
    public Sample randomSample() {
        return nn.randomTestSample();
    }

    // ------------------------------------------------------------------ validare

    private static double[] input(double[] in) {
        if (in == null) return new double[NeuralNetworkService.INPUTS];
        if (in.length != NeuralNetworkService.INPUTS) {
            throw new IllegalArgumentException("Desenul trebuie să aibă exact " + NeuralNetworkService.INPUTS + " pixeli");
        }
        double[] out = new double[in.length];
        for (int i = 0; i < in.length; i++) {
            double v = in[i];
            out[i] = Double.isFinite(v) ? Math.max(0, Math.min(1, v)) : 0;
        }
        return out;
    }

    private static double learningRate(Double lr) {
        if (lr == null) return 0.05;
        if (!(lr > 0 && lr <= 1)) throw new IllegalArgumentException("Rata de învățare trebuie să fie în (0, 1]");
        return lr;
    }

    private static int clamp(int v, int lo, int hi) {
        return Math.max(lo, Math.min(hi, v));
    }
}
