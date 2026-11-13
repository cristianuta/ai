package ro.lab.ai.nn;

import java.util.List;

/** Obiectele JSON schimbate cu front-end-ul pentru rețeaua neurală. */
public final class NnApi {

    private NnApi() {
    }

    public record InputRequest(double[] input) {}

    public record TrainRequest(Integer batches, Double learningRate, double[] input) {}

    public record ResetRequest(int[] hidden, double[] input) {}

    public record DataRequest(Double noise, double[] input) {}

    public record SampleRequest(double[] pixels, Integer label) {}

    public record Sample(double[] pixels, int label) {}

    public record ActivationsResponse(float[][] activations) {}

    public record History(List<Double> trainLoss, List<Double> testLoss,
                          List<Double> trainAccuracy, List<Double> testAccuracy) {}

    /** Tot ce are nevoie interfața ca să deseneze rețeaua. */
    public record Snapshot(
            int[] sizes,
            float[][] weights,
            float[][] biases,
            float[][] activations,
            int epoch,
            Double lastLoss,
            int trainSamples,
            int testSamples,
            int customSamples,
            int parameters,
            double noise,
            History history) {}
}
