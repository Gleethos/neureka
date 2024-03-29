package testutility;

import java.util.function.Supplier;

public class Sleep {

    public static boolean until(int delay, Supplier<Boolean> condition) {
        return until(delay, 10, condition);
    }

    public static boolean until(int delay, int intermediate, Supplier<Boolean> condition) {
        long start = System.currentTimeMillis();
        while ( System.currentTimeMillis() - start < delay ) {
            if ( condition.get() ) return true;
            try {
                Thread.sleep(intermediate); // Text book busy waiting. But the alternative is waiting the entire delay.
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        return false;
    }

}
