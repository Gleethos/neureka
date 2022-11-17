package testutility;

import neureka.Tsr;
import neureka.math.Function;

public class Measure {

    public static void measure(String taskName, int warmup, Runnable task) {
        for ( int i = 0; i < warmup; i++ )
            task.run();

        System.out.println("It took "+seconds(task)+" seconds to complete "+taskName+"!");
    }

    public static double seconds(Runnable task) {
        final long OVERHEAD = 1100; // (Can be up to 1500)
        long startTime = System.nanoTime();
        task.run();
        long endTime = System.nanoTime() ;
        return ( (( endTime - startTime ) - OVERHEAD) / 1_000_000_000.0 );
    }

    public static double minutes(Runnable task) {
        return seconds(task) / 60d;
    }

    /**
     *  Use this to test performance and do some tuning.
     */
    public static void main(String... args) {
        System.out.println(averageSeconds(185, _fun64())+"s");
        System.out.println(averageSeconds(185, _funi64())+"s");
        //System.out.println(averageSeconds(35, __fun64())+"s");
        //System.out.println(averageSeconds(35, _f64())+"s");
        System.out.println("DONE!");
        System.exit(0);
    }

    public static double averageSeconds(int times, Runnable task)
    {
        for ( int i = 0; i < Math.min(times, 1_500); i++ ) task.run();

        double average = 0;
        for ( int i = 0; i < times; i++ ) average += seconds(task);
        average /= times;
        return average;
    }

    private static Runnable _f32() {
        Tsr<Float> a = Tsr.ofFloats().withShape(40, 300, 200).andFill(-1f, 5f, 0.7f, 9f, -14.75f);
        Tsr<Float> b = Tsr.ofFloats().withShape(40, 300, 200).andWhere((i, indices) -> (float)((Math.pow(7,i)%11)-5));
        Tsr<Float> c = Tsr.ofFloats().withShape(40, 300, 200).andSeed("I am a happy seed! :D");
        Function f = Function.of("relu(i0+i1*3)*i2");
        return  ()-> f.call(a, b, c);
    }

    private static Runnable _f64() {
        Tsr<Double> a = Tsr.ofDoubles().withShape(40, 300, 200).andFill(-1d, 5d, 0.7d, 9d, -14.75d);
        Tsr<Double> b = Tsr.ofDoubles().withShape(40, 300, 200).andWhere((i, indices) -> ((Math.pow(7,i)%11)-5));
        Tsr<Double> c = Tsr.ofDoubles().withShape(40, 300, 200).andSeed("I am a happy seed! :D");
        Function f = Function.of("relu(i0+i1*3)*i2");
        return  ()-> f.call(a, b, c);
    }

    private static Runnable _fun64() {
        Tsr<Double> a = Tsr.ofDoubles().withShape(42, 300, 200).andFill(-1d, 5d, 0.7d, 9d, -14.75d);
        Function f = Function.of("tanh(i0)");
        return  ()-> f.call(a);
    }
    private static Runnable _funi64() {
        Tsr<Double> a = Tsr.ofDoubles().withShape(42, 300, 200).andFill(-1d, 5d, 0.7d, 9d, -14.75d);
        Function f = Function.of("fast_tanh(i0)");
        return  ()-> f.call(a);
    }

    private static Runnable __fun64() {
        double[] a = new double[42*300*200];
        double[] data = {-1d, 5d, 0.7d, 9d, -14.75d};
        for ( int i = 0; i < a.length; i++ ) a[i] = data[i%data.length];
        return  ()-> {
            double[] out = new double[a.length];
            for ( int i = 0; i < a.length; i++ ) out[i] = Math.tanh(a[i]);
        };
    }

    private static Runnable _random1() {
        return  ()-> {
            Tsr.ofRandom(Double.class, 40, 300, 200);
        };
    }

    private static Runnable _random2() {
        Function f = Function.of("random(I[0])");
        return  ()-> {
            Tsr<Double> a = Tsr.ofDoubles().withShape(40, 300, 200).all(0d);
            f.call(a);
        };
    }

    private static Runnable _map() {
        Tsr<Double> a = Tsr.ofDoubles().withShape(40, 300, 200).andFill(-1d, 5d, 0.7d, 9d, -14.75d);
        return  ()-> a.mapTo( Float.class, Double::floatValue );
    }

}
