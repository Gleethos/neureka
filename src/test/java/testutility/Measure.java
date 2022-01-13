package testutility;

import neureka.Tsr;
import neureka.calculus.Function;

public class Measure {

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

    public static void main(String... args) {

        System.out.println(seconds(_f32())+"s");
        System.out.println(seconds(_f64())+"s");
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

}
