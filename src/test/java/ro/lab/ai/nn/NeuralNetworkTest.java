package ro.lab.ai.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;

import org.junit.jupiter.api.Test;

class NeuralNetworkTest {

    @Test
    void outputIsAProbabilityDistribution() {
        NeuralNetwork net = new NeuralNetwork(new int[]{64, 16, 10}, new Random(1));
        double[] out = net.forward(DigitDataset.template(3))[2];
        double sum = 0;
        for (double p : out) {
            assertTrue(p > 0 && p < 1);
            sum += p;
        }
        assertEquals(1.0, sum, 1e-9);
    }

    @Test
    void parameterCountMatchesArchitecture() {
        NeuralNetwork net = new NeuralNetwork(new int[]{64, 16, 16, 10}, new Random(1));
        assertEquals(64 * 16 + 16 + 16 * 16 + 16 + 16 * 10 + 10, net.parameterCount());
    }

    @Test
    void backpropagationGradientMatchesNumericalGradient() {
        // Cu lr = 1, momentum = 0, l2 = 0 și un singur exemplu, schimbarea unei ponderi = -gradientul ei.
        Random rnd = new Random(7);
        double[] x = DigitDataset.augment(DigitDataset.template(5), 0.1, rnd);
        int y = 5;
        int layer = 0, index = 3 * 64 + 20;

        NeuralNetwork a = new NeuralNetwork(new int[]{64, 8, 6, 10}, new Random(42));
        NeuralNetwork b = new NeuralNetwork(new int[]{64, 8, 6, 10}, new Random(42));
        double eps = 1e-6;
        b.weights()[layer][index] += eps;
        double numerical = (loss(b, x, y) - loss(a, x, y)) / eps;

        double before = a.weights()[layer][index];
        a.trainBatch(new double[][]{x}, new int[]{y}, 1.0, 0, 0);
        double analytical = before - a.weights()[layer][index];

        assertEquals(numerical, analytical, 1e-4);
    }

    @Test
    void learnsToRecognizeDigits() {
        Random rnd = new Random(123);
        DigitDataset.Data train = DigitDataset.generate(60, 0.1, rnd);
        DigitDataset.Data test = DigitDataset.generate(20, 0.1, rnd);
        NeuralNetwork net = new NeuralNetwork(new int[]{64, 16, 16, 10}, rnd);

        double initial = net.evaluate(test.xs(), test.ys()).accuracy();
        for (int epoch = 0; epoch < 20; epoch++) {
            for (int start = 0; start < train.size(); start += 16) {
                int n = Math.min(16, train.size() - start);
                double[][] xs = new double[n][];
                int[] ys = new int[n];
                for (int k = 0; k < n; k++) {
                    int idx = rnd.nextInt(train.size());
                    xs[k] = train.xs().get(idx);
                    ys[k] = train.ys().get(idx);
                }
                net.trainBatch(xs, ys, 0.05, 0.9, 1e-4);
            }
        }
        double accuracy = net.evaluate(test.xs(), test.ys()).accuracy();
        assertTrue(initial < 0.4, "rețeaua neantrenată ar trebui să ghicească: " + initial);
        assertTrue(accuracy > 0.8, "acuratețe prea mică după antrenare: " + accuracy);
    }

    private static double loss(NeuralNetwork net, double[] x, int y) {
        return -Math.log(net.predict(x).probabilities()[y]);
    }
}
