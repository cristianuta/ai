package ro.lab.ai.nn;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

import org.springframework.stereotype.Service;
import org.springframework.web.context.annotation.SessionScope;

import ro.lab.ai.nn.NnApi.History;
import ro.lab.ai.nn.NnApi.Sample;
import ro.lab.ai.nn.NnApi.Snapshot;

/**
 * Starea laboratorului de rețele neurale pentru un utilizator (o sesiune HTTP):
 * rețeaua, datele de antrenare/test, istoricul antrenării.
 */
@Service
@SessionScope
public class NeuralNetworkService {

    public static final int INPUTS = DigitDataset.PIXELS;
    public static final int OUTPUTS = 10;
    public static final int BATCH_SIZE = 16;
    static final int TRAIN_PER_CLASS = 80;
    static final int TEST_PER_CLASS = 25;
    static final int VARIATIONS_PER_CUSTOM_SAMPLE = 20;
    private static final double MOMENTUM = 0.9;
    private static final double L2 = 1e-4;

    private final Random rnd = new Random();
    private final List<Sample> customSamples = new ArrayList<>();
    private final List<Double> trainLoss = new ArrayList<>();
    private final List<Double> testLoss = new ArrayList<>();
    private final List<Double> trainAccuracy = new ArrayList<>();
    private final List<Double> testAccuracy = new ArrayList<>();

    private double noise = 0.15;
    private NeuralNetwork network;
    private DigitDataset.Data train;
    private DigitDataset.Data test;
    private int[] order;
    private int cursor;
    private int epoch;
    private Double lastLoss;

    public NeuralNetworkService() {
        buildData();
        reset(new int[]{16, 16});
    }

    /** Creează o rețea nouă (ponderi aleatoare) cu straturile ascunse date. */
    public synchronized void reset(int[] hidden) {
        int[] sizes = new int[hidden.length + 2];
        sizes[0] = INPUTS;
        System.arraycopy(hidden, 0, sizes, 1, hidden.length);
        sizes[sizes.length - 1] = OUTPUTS;
        network = new NeuralNetwork(sizes, rnd);
        epoch = 0;
        lastLoss = null;
        trainLoss.clear();
        testLoss.clear();
        trainAccuracy.clear();
        testAccuracy.clear();
        shuffleOrder();
        recordEpoch();
    }

    /** Generează alte exemple de antrenare/test cu nivelul de zgomot dat. */
    public synchronized void regenerateData(double noise) {
        this.noise = noise;
        buildData();
        recordEpoch();
    }

    public synchronized void train(int batches, double learningRate) {
        for (int b = 0; b < batches; b++) runBatch(learningRate);
    }

    public synchronized void trainEpoch(double learningRate) {
        int start = epoch;
        while (epoch == start) runBatch(learningRate);
    }

    /** Adaugă desenul utilizatorului (plus variații) în setul de antrenare. */
    public synchronized void addSample(double[] pixels, int label) {
        Sample sample = new Sample(pixels.clone(), label);
        customSamples.add(sample);
        addToTrain(sample);
        shuffleOrder();
    }

    public synchronized Sample randomTestSample() {
        int k = rnd.nextInt(test.size());
        return new Sample(test.xs().get(k).clone(), test.ys().get(k));
    }

    public synchronized float[][] activations(double[] input) {
        return toFloat(network.forward(input));
    }

    public synchronized Snapshot snapshot(double[] input) {
        return new Snapshot(
                network.sizes(),
                toFloat(network.weights()),
                toFloat(network.biases()),
                toFloat(network.forward(input)),
                epoch,
                lastLoss,
                train.size(),
                test.size(),
                customSamples.size(),
                network.parameterCount(),
                noise,
                new History(List.copyOf(trainLoss), List.copyOf(testLoss),
                        List.copyOf(trainAccuracy), List.copyOf(testAccuracy)));
    }

    // ------------------------------------------------------------------ intern

    private void buildData() {
        train = DigitDataset.generate(TRAIN_PER_CLASS, noise, rnd);
        test = DigitDataset.generate(TEST_PER_CLASS, noise, rnd);
        customSamples.forEach(this::addToTrain);
        shuffleOrder();
    }

    private void addToTrain(Sample s) {
        for (int k = 0; k < VARIATIONS_PER_CUSTOM_SAMPLE; k++) {
            train.xs().add(k == 0 ? s.pixels().clone() : DigitDataset.augment(s.pixels(), 0.08, rnd));
            train.ys().add(s.label());
        }
    }

    private void shuffleOrder() {
        order = IntStream.range(0, train.size()).toArray();
        shuffle(order);
        cursor = 0;
    }

    private void shuffle(int[] a) {
        for (int i = a.length - 1; i > 0; i--) {
            int j = rnd.nextInt(i + 1);
            int t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
    }

    private void runBatch(double learningRate) {
        int end = Math.min(cursor + BATCH_SIZE, order.length);
        int n = end - cursor;
        double[][] xs = new double[n][];
        int[] ys = new int[n];
        for (int k = 0; k < n; k++) {
            xs[k] = train.xs().get(order[cursor + k]);
            ys[k] = train.ys().get(order[cursor + k]);
        }
        network.trainBatch(xs, ys, learningRate, MOMENTUM, L2);
        cursor = end;
        if (cursor >= order.length) {
            epoch++;
            shuffle(order);
            cursor = 0;
            recordEpoch();
        }
    }

    private void recordEpoch() {
        NeuralNetwork.Evaluation tr = network.evaluate(train.xs(), train.ys());
        NeuralNetwork.Evaluation te = network.evaluate(test.xs(), test.ys());
        trainLoss.add(tr.loss());
        testLoss.add(te.loss());
        trainAccuracy.add(tr.accuracy() * 100);
        testAccuracy.add(te.accuracy() * 100);
        lastLoss = tr.loss();
    }

    private static float[][] toFloat(double[][] a) {
        float[][] out = new float[a.length][];
        for (int l = 0; l < a.length; l++) {
            out[l] = new float[a[l].length];
            for (int k = 0; k < a[l].length; k++) out[l][k] = (float) a[l][k];
        }
        return out;
    }
}
