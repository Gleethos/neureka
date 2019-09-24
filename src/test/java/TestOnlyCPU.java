import neureka.core.T;
import neureka.core.function.IFunction;
import neureka.core.function.factory.assembly.FunctionBuilder;
import org.junit.Test;
import util.NTester_Tensor;

public class TestOnlyCPU {

    @Test
    public void testTest() throws InterruptedException {

        NTester_Tensor tester = new NTester_Tensor("Tensor tester (only cpu)");

        T x = new T(new int[]{1}, 3).setRqsGradient(true);
        T b = new T(new int[]{1}, -4);
        T w = new T(new int[]{1}, 2);
        /**
         *      ((3-4)*2)^2 = 4
         *  dx:   8*3 - 32  = -8
         * */
        T y = new T(new T[]{x, b, w}, "((i0+i1)*i2)^2");
        tester.testTensor(y, new String[]{"[1]:(4.0); ->d[1]:(-8.0), "});
        y.backward(new T(2));
        tester.testTensor(x, new String[]{"-16.0"});

        y = new T("(","(",x,"+",b,")","*",w,")^2");
        tester.testTensor(y, new String[]{"[1]:(4.0); ->d[1]:(-8.0), "});
        y.backward(new T(1));
        tester.testTensor(x, new String[]{"-8.0"});

        y = new T("((",x,"+",b,")*",w,")^2");
        tester.testTensor(y, new String[]{"[1]:(4.0); ->d[1]:(-8.0), "});
        y.backward(new T(-1));
        tester.testTensor(x, new String[]{"8.0"});
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
        tester.testTensor(z, new String[]{"[2x1x2]:(19.0, 22.0, 1.0, -6.0)"});

        z = new T(new Object[]{x, "x", y});
        tester.testTensor(z, new String[]{"[2x1x2]:(19.0, 22.0, 1.0, -6.0)"});
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

        tester.testTensor(y, new String[]{":g:(null)"});
        z = new T(new T[]{x, y}, "I0xi1");
        tester.testTensor(z, new String[]{"[2x2]:(15.0, 15.0, 18.0, 8.0)"});

        z = new T(new Object[]{x, "x", y});
        tester.testTensor(z, new String[]{"[2x2]:(15.0, 15.0, 18.0, 8.0)"});

        z.backward(new T(new int[]{2, 2}, 1));
        tester.testTensor(y, new String[]{"[2x2]:(-1.0, 3.0, 2.0, 3.0):g:(6.0, 9.0, 4.0, 9.0)"});
        System.out.println(z);


        Thread.sleep(6000);
    }


    @Test
    public void testAD()
    {
        NTester_Tensor tester = new NTester_Tensor("Tensor tester (only cpu)");

        T x = new T(
                new int[]{2, 2},
                new double[]{
                        -1, 2,
                        -3, 3,
                }
        ).setRqsGradient(true);
        T y = new T(new int[]{1, 1}, -3);
        T z = new T(new T[]{x, y}, "I0xi1");
        tester.testTensor(z, new String[]{"[2x2]:(3.0, -6.0, 9.0, -9.0)"});
        z.backward(new T(new int[]{2, 2}, 1));
        tester.testTensor(x, new String[]{"[2x2]:(-1.0, 2.0, -3.0, 3.0):g:(-3.0, -3.0, -3.0, -3.0)"});
        //---
        x = new T(new int[]{1}, 0.1).setRqsGradient(true);
        IFunction tanh = FunctionBuilder.build("tanh(i0)", true);
        IFunction tenxx = FunctionBuilder.build("i0*100", true);
        z = tenxx.activate(new T[]{tanh.activate(new T[]{x})});
        tester.testTensor(z, new String[]{"[1]:(9.950371902099892)"});
        z.backward(new T(new int[]{1}, 1));
        tester.testTensor(x, new String[]{"[1]:(0.1):g:(99.00990099009901)"});
        tester.testTensor(z, new String[]{"[1]:(9.950371902099892); ->d[1]:(99.00990099009901), "});
        //---
        tester.testContains(
                z.toString("dgc"),
                new String[]{"[1]:(9,95037E0); ->d[1]:(99,0099E0), "},
                "test double formatting"
        );

    }




}
