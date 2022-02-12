package testutility;

import java.util.Arrays;

public class Statistics {

    private final double[] data;
    private final int size;
    private Double mean = null;
    private Double variance = null;

    public Statistics(double[] data) {
        this.data = data;
        size = data.length;
    }

    public double getMean() {
        if ( this.mean != null ) return mean;
        double sum = 0.0;
        for(double a : data)
            sum += a;

        this.mean = sum/size;
        return this.mean;
    }

    public double getVariance() {
        if ( this.variance != null ) return variance;
        double mean = getMean();
        double temp = 0;
        for(double a :data)
            temp += (a-mean)*(a-mean);
        this.variance = temp/(size-1);
        return this.variance;
    }

    double getStdDev() {
        return Math.sqrt(getVariance());
    }

    public double median() {
        Arrays.sort(data);
        if (data.length % 2 == 0)
            return (data[(data.length / 2) - 1] + data[data.length / 2]) / 2.0;
        return data[data.length / 2];
    }
}
