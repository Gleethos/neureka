import neureka.core.T;
import org.junit.Assert;
import org.junit.Test;
import util.NTester;

public class TestOnlyCPU {

    @Test
    public void testTest() throws InterruptedException {

        NTester tester = new NTester("helper", false);

        T x = new T(new int[]{1}, 3).setRqsGradient(true);
        T b = new T(new int[]{1}, -4);
        T w = new T(new int[]{1}, 2);
        /**
         *      ((3-4)*2)^2 = 4
         *  dx:   8*3 - 32  = -8
         * */
        T y = new T(new T[]{x, b, w}, "((i0+i1)*i2)^2");
        Assert.assertEquals("[1]:(4.0); ->d[1]:(-8.0), ", y.toString());
        y.backward(new T(2));
        Assert.assertEquals(tester.stringified(new double[]{-16}), tester.stringified(x.gradient()));

        Thread.sleep(6000);
    }



}
