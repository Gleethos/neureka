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

        y = new T("(","(",x,"+",b,")","*",w,")^2");
        Assert.assertEquals("[1]:(4.0); ->d[1]:(-8.0), ", y.toString());
        y.backward(new T(1));
        Assert.assertEquals(tester.stringified(new double[]{-8}), tester.stringified(x.gradient()));


        y = new T("((",x,"+",b,")*",w,")^2");
        Assert.assertEquals("[1]:(4.0); ->d[1]:(-8.0), ", y.toString());
        y.backward(new T(-1));
        Assert.assertEquals(tester.stringified(new double[]{8}), tester.stringified(x.gradient()));


        //===========================================
        x = new T(
                new int[]{2, 3, 1},
                new double[]{
                        3, 2,
                        -1, -2,
                        2, 4
                }
        );
        y = new T(
                new int[]{1, 3, 2},
                new double[]{
                        4, -1, 3,
                        2, 3, -1
                });
        T z = new T(new T[]{x, y}, "I0xi1");
        Assert.assertEquals(z.toString().contains("[2x1x2]:(19.0, 22.0, 1.0, -6.0)"), true);

        z = new T(new Object[]{x, "x", y});
        Assert.assertEquals(z.toString().contains("[2x1x2]:(19.0, 22.0, 1.0, -6.0)"), true);
        //=======================
        x = new T(
                new int[]{3, 3},
                new double[]{
                        1, 2, 5,
                        -1, 4, -2,
                        -2, 3, 4,
                }
        );
        y = new T(
                new int[]{2, 2},
                new double[]{
                        -1, 3,
                        2, 3,
                }).setRqsGradient(true);

        Assert.assertEquals(y.toString().contains(":g:(null)"), true);
        z = new T(new T[]{x, y}, "I0xi1");
        Assert.assertEquals(z.toString().contains("[2x2]:(15.0, 15.0, 18.0, 8.0)"), true);

        z = new T(new Object[]{x, "x", y});
        Assert.assertEquals(z.toString().contains("[2x2]:(15.0, 15.0, 18.0, 8.0)"), true);
        z.backward(new T(new int[]{2, 2}, 1));
        Assert.assertEquals(y.toString().contains("[2x2]:(-1.0, 3.0, 2.0, 3.0):g:(6.0, 9.0, 4.0, 9.0)"), true);
        System.out.println(z);

        Thread.sleep(6000);
    }



}
