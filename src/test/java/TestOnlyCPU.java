import neureka.core.Tsr;
import neureka.core.function.IFunction;
import neureka.core.function.factory.assembly.FunctionBuilder;
import org.junit.Test;
import util.NTester_Tensor;

public class TestOnlyCPU {

    @Test
    public void testTest() throws InterruptedException {

        NTester_Tensor tester = new NTester_Tensor("Tensor tester (only cpu)");

        Tsr x = new Tsr(new int[]{1}, 3).setRqsGradient(true);
        Tsr b = new Tsr(new int[]{1}, -4);
        Tsr w = new Tsr(new int[]{1}, 2);
        /**
         *      ((3-4)*2)^2 = 4
         *  dx:   8*3 - 32  = -8
         * */
        Tsr y = new Tsr(new Tsr[]{x, b, w}, "((i0+i1)*i2)^2");
        tester.testTensor(y, new String[]{"[1]:(4.0); ->d[1]:(-8.0), "});
        y.backward(new Tsr(2));
        tester.testTensor(x, new String[]{"-16.0"});

        y = new Tsr("(","(",x,"+",b,")","*",w,")^2");
        tester.testTensor(y, new String[]{"[1]:(4.0); ->d[1]:(-8.0), "});
        y.backward(new Tsr(1));
        tester.testTensor(x, new String[]{"-8.0"});

        y = new Tsr("((",x,"+",b,")*",w,")^2");
        tester.testTensor(y, new String[]{"[1]:(4.0); ->d[1]:(-8.0), "});
        y.backward(new Tsr(-1));
        tester.testTensor(x, new String[]{"8.0"});
        //===========================================
        x = new Tsr(
                new int[]{2, 3, 1},
                new double[]{
                        3, 2,
                        -1, -2,
                        2, 4
                }
        );
        y = new Tsr(
                new int[]{1, 3, 2},
                new double[]{
                        4, -1, 3,
                        2, 3, -1
                });
        Tsr z = new Tsr(new Tsr[]{x, y}, "I0xi1");
        tester.testTensor(z, new String[]{"[2x1x2]:(19.0, 22.0, 1.0, -6.0)"});

        z = new Tsr(new Object[]{x, "x", y});
        tester.testTensor(z, new String[]{"[2x1x2]:(19.0, 22.0, 1.0, -6.0)"});
        //=======================
        x = new Tsr(
                new int[]{3, 3},
                new double[]{
                        1, 2, 5,
                        -1, 4, -2,
                        -2, 3, 4,
                }
        );
        y = new Tsr(
                new int[]{2, 2},
                new double[]{
                        -1, 3,
                        2, 3,
                }).setRqsGradient(true);

        tester.testTensor(y, new String[]{":g:(null)"});
        z = new Tsr(new Tsr[]{x, y}, "I0xi1");
        tester.testTensor(z, new String[]{"[2x2]:(15.0, 15.0, 18.0, 8.0)"});

        z = new Tsr(new Object[]{x, "x", y});
        tester.testTensor(z, new String[]{"[2x2]:(15.0, 15.0, 18.0, 8.0)"});

        z.backward(new Tsr(new int[]{2, 2}, 1));
        tester.testTensor(y, new String[]{"[2x2]:(-1.0, 3.0, 2.0, 3.0):g:(6.0, 9.0, 4.0, 9.0)"});
        System.out.println(z);
        //====
        x = new Tsr(new int[]{1}, 3);
        b = new Tsr(new int[]{1}, -5);
        w = new Tsr(new int[]{1}, -2);
        z = new Tsr(new Tsr[]{x, b, w}, "I0*i1*i2");
        tester.testTensor(z, new String[]{"[1]:(30.0)"});

        x = new Tsr(new int[]{1}, 4).setRqsGradient(true);
        b = new Tsr(new int[]{1}, 0.5);
        w = new Tsr(new int[]{1}, 0.5);
        y = new Tsr(new Tsr[]{x, b, w}, "(2^i0^i1^i2^2");
        tester.testTensor(y, new String[]{"[1]:(4.0);", " ->d[1]:(1.38629E0), "});
        //===
        Thread.sleep(1000);
        tester.closeWindows();
    }


    @Test
    public void testAD()
    {
        NTester_Tensor tester = new NTester_Tensor("Tensor tester (only cpu)");

        Tsr x = new Tsr(
                new int[]{2, 2},
                new double[]{
                        -1, 2,
                        -3, 3,
                }
        ).setRqsGradient(true);
        Tsr y = new Tsr(new int[]{1, 1}, -3);
        Tsr z = new Tsr(new Tsr[]{x, y}, "I0xi1");
        tester.testTensor(z, new String[]{"[2x2]:(3.0, -6.0, 9.0, -9.0)"});
        z.backward(new Tsr(new int[]{2, 2}, 1));
        tester.testTensor(x, new String[]{"[2x2]:(-1.0, 2.0, -3.0, 3.0):g:(-3.0, -3.0, -3.0, -3.0)"});
        //---
        x = new Tsr(new int[]{1}, 0.1).setRqsGradient(true);
        IFunction tanh = FunctionBuilder.build("tanh(i0)", true);
        IFunction tenxx = FunctionBuilder.build("i0*100", true);
        z = tenxx.activate(new Tsr[]{tanh.activate(new Tsr[]{x})});
        tester.testTensor(z, new String[]{"[1]:(9.95037E0)"});
        z.backward(new Tsr(new int[]{1}, 1));
        tester.testTensor(x, new String[]{"[1]:(0.1):g:(99.0099E0)"});
        tester.testTensor(z, new String[]{"[1]:(9.95037E0); ->d[1]:(99.0099E0), "});
        //---
        tester.testContains(
                z.toString("dgc"),
                new String[]{"[1]:(9.95037E0); ->d[1]:(99.0099E0),"},
                "test double formatting"
        );
        tester.testContains(
                new Tsr(3).toString("dgc"),
                new String[]{"[1]:(3.0)"},
                "test FP formatting"
        );

        try {
            Thread.sleep(6000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        tester.closeWindows();
    }




}
