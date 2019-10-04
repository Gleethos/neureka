
import neureka.core.Tsr;
import neureka.core.device.Device;
import neureka.core.function.IFunction;
import org.junit.Test;
import util.NTester_Function;
import util.NTester_Tensor;
import util.NTester_TensorDevice;

public class TestBroad {

    @Test
    public void testTensorFunctions() {

        NTester_Function tester = new NTester_Function("Testing function factory and scalar calculations");
        //EXPRESSION TESTING:
        tester.testExpression("ig0*(igj)xI[g1]", "((Ig[0]*Ig[j])xIg[1])", "");
        tester.testExpression("sum(ij)", "sum(I[j])", "");
        tester.testExpression("sum(1*(4-2/ij))", "sum(1.0*(4.0-(2.0/I[j])))", "");
        tester.testExpression("quadratic(ligmoid(Ij))", "quad(lig(I[j]))", "");
        tester.testExpression("softplus(I[3]^(3/i1)/sum(Ij^2)-23+I0/i1)", "lig((((I[3]^(3.0/I[1]))/sum(I[j]^2.0))-23.0)+(I[0]/I[1]))", "");
        tester.testExpression("1+3+5-23+I0*45/(345-651^I3-6)", "(1.0+3.0+(5.0-23.0)+(I[0]*(45.0/(345.0-(651.0^I[3])-6.0))))", "");
        tester.testExpression("sin(23*i1)-cos(i0^0.3)+tanh(23)", "((sin(23.0*I[1])-cos(I[0]^0.3))+tanh(23.0))", "");
        tester.testExpression("2*3/2-1", "((2.0*(3.0/2.0))-1.0)", "");
        tester.testExpression("3x5xI[4]xI[3]", "(((3.0x5.0)xI[4])xI[3])", "");
        tester.testExpression("[1,0, 5,3, 4]:(tanh(i0xi1))", "([1,0,5,3,4]:(tanh(I[0]xI[1])))", "");
        tester.testExpression("[0,2, 1,3, -1](sig(I0))", "([0,2,1,3,-1]:(sig(I[0])))", "");
        tester.testExpression("I[0]<-I[1]->I[2]", "((I[0]<-I[1])->I[2])", "");
        tester.testExpression("quadratic(I[0]) -> I[1] -> I[2]", "((quad(I[0])->I[1])->I[2])", "");

        //ACTIVATION TESTING:
        double[] input1 = {};
        tester.testActivation("6/2*(1+2)", input1, 9, "");
        input1 = new double[]{2, 3.2, 6};
        tester.testActivation("sum(Ij)", input1, 11.2, "");
        input1 = new double[]{0.5, 0.5, 100};
        tester.testActivation("prod(Ij)", input1, 25, "");
        input1 = new double[]{0.5, 0.5, 10};
        tester.testActivation("prod(prod(Ij))", input1, (2.5 * 2.5 * 2.5), "");
        input1 = new double[]{5, 4, 3, 12};//12/4-5+2+3
        tester.testActivation("I3/i[1]-I0+2+i2", input1, (3), "");
        input1 = new double[]{-4, -2, 6, -3, -8};//-3*-2/(-8--4-2)
        tester.testActivation("i3*i1/(i4-i0-2)-sig(0)+tanh(0)", input1, (-1.5), "");
        input1 = new double[]{2, 3, -2};//-3*-2/(-8--4-2)
        tester.testDeriviation("(i0*i1)*i2", input1, 0, (-6), "");
        input1 = new double[]{2, 3, -2};//-3*-2/(-8--4-2)
        tester.testDeriviation("lig(i0*i1)*i2", input1, 0, (-5.985164261060192), "");
        input1 = new double[]{2, 3, -2};//-3*-2/(-8--4-2)
        tester.testDeriviation("prod(ij)", input1, 1, (-4), "");

        Tsr[] tsrs = new Tsr[]{new Tsr(new int[]{2}, new double[]{1, 2}), new Tsr(new int[]{2},new double[]{3, -4})};
        Tsr expected = new Tsr(new int[]{2}, new double[]{0.9701425001453319, -0.8944271909999159});
        tester.testActivation("tanh(sum(Ij))", tsrs, expected, "");

        expected = new Tsr(new int[]{2}, new double[]{0.31326168751822286, 0.6931471805599453});
        tester.testActivation("lig(prod(Ij-2))", tsrs, expected, "");

        tsrs = new Tsr[]{new Tsr(new int[]{2, 4}, new double[]{10, 12, 16, 21, 33, 66, 222, 15})};
        expected = new Tsr(new int[]{1, 2, 2, 2}, new double[]{8.000335406372896, 10.000045398899216, 14.000000831528373, 19.000000005602796, 31.000000000000032, 64.0, 220.0, 13.000002260326852});
        tester.testActivation("lig([-1, 0, -2, -2](Ij-2))", tsrs, expected, "");

        tsrs = new Tsr[]{
                new Tsr(new int[]{2}, new double[]{-1, 3}),
                new Tsr(new int[]{2}, new double[]{7, -1}),
                new Tsr(new int[]{2}, new double[]{2, 2}),
        };
        expected = new Tsr(new int[]{2}, new double[]{-0.0018221023888012912, 0.2845552390654007});
        tester.testDerivative("lig(i0*i1)*i2", tsrs, 1, expected, "");
        try {
            Thread.sleep(2000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        tester.closeWindows();
    }

    @Test
    public void testTensorCore() {

        NTester_Tensor tester = new NTester_Tensor("Testing core tensor functionality");
        //---
        Tsr tensor1 = new Tsr(3).setRqsGradient(true);
        Tsr tensor2 = new Tsr(-4);
        Tsr tensor3 = new Tsr(2);
        tester.testInjection(
                new Tsr[]{tensor1, tensor2, tensor3},
                "(Ig[0]<-I[1])->I[2]",
                new String[][]{
                        {"empty"},//result
                        {"(3.0)", "g:(-4.0)"},//tensor1
                        {"(-4.0)"},//tensor2
                        {"(-4.0)"},//tensor3
                });
        tensor1 = new Tsr(3).setRqsGradient(true);
        tensor2 = new Tsr(-4);
        tensor3 = new Tsr(2);
        Tsr result = IFunction.setup.commit(new Tsr[]{tensor1, tensor2, tensor3}, "(Ig[0]<-I[1])->I[2]", true);
        tester.testContains(
                result.toString(),
                new String[]{"(-4.0)"},
                "Testing if IFunction.setup.commit() returns non unique result!"
        );
        //---
        tensor1 = new Tsr(new int[]{1, 3}, 2);
        tensor2 = new Tsr(new int[]{2, 1}, -1);
        tensor1.setRqsGradient(true);
        tensor2.setRqsGradient(true);
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1, tensor2},
                "relu(I[0]xI[1])",
                new String[]{
                        "[2x3]:(-0.02, -0.02, -0.02, -0.02, -0.02, -0.02);",
                            " =>d|[ [2x3]:(0.01, 0.01, 0.01, 0.01, 0.01, 0.01) ]|:t{ [2x3]:(-2.0, -2.0, -2.0, -2.0, -2.0, -2.0);",
                            " =>d|[ [2x1]:(-1.0, -1.0) ]|:t{ [1x3]:(2.0, 2.0, 2.0) },",
                            " =>d|[ [1x3]:(2.0, 2.0, 2.0) ]|:t{ [2x1]:(-1.0, -1.0) },",
                        "  }, "
                });
        //---
        tensor1 = new Tsr(new int[]{1, 3}, 2);
        tensor2 = new Tsr(new int[]{2, 1}, -1);
        tensor1.setRqsGradient(true);
        tensor2.setRqsGradient(true);
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1, tensor2},
                "lig((I[0]xI[1])*-100)",
                new String[]{
                        "[2x3]:(200.0, 200.0, 200.0, 200.0, 200.0, 200.0);",
                        " =>d|[ [2x3]:(-100.0, -100.0, -100.0, -100.0, -100.0, -100.0) ]|:t{ [2x3]:(-2.0, -2.0, -2.0, -2.0, -2.0, -2.0);",
                        " =>d|[ [1x3]:(2.0, 2.0, 2.0) ]|:t{ [2x1]:(-1.0, -1.0) },",
                        " =>d|[ [2x1]:(-1.0, -1.0) ]|:t{ [1x3]:(2.0, 2.0, 2.0) },",
                        "  }, "
                }
        );
        //---
        tensor1 = new Tsr(new int[]{2, 3, 4}, 2);
        tensor1.setRqsGradient(false);
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1},//, tensor2},/<=TODO make this throw an exception (if input does not match function)
                "lig([-2, 1, 0, -2]:(I[0])*-100)",
                new String[]{"[2x3x2x2]:(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)"}
        );
        //---
        tensor1 = new Tsr(new int[]{2}, 2);
        tensor2 = new Tsr(new int[]{2}, 4);
        tensor1.setRqsGradient(true);
        tensor2.setRqsGradient(true);
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1, tensor2},
                "lig(tanh(I[0]*I[1]*2)*I[1])",
                new String[]{
                        "[2]:(4.010500886001868, 4.010500886001868); ",
                        "=>d|[ [2]:(3.9275027410108176, 3.9275027410108176) ]|" +
                                ":t{ [2]:(0.9980525784828885, 0.9980525784828885); ",
                        "=>d|[ [2]:(0.015564202334630739, 0.015564202334630739) ]|:t{ [2]:(4.0, 4.0) }, ",
                        "=>d|[ [2]:(0.031128404669261478, 0.031128404669261478) ]|:t{ [2]:(2.0, 2.0) }, ",
                        "}, ",
                        "=>d|[ [2]:(0.9799635594161147, 0.9799635594161147) ]|" +
                                ":t{ [2]:(4.0, 4.0) }, "
                }
        );
        //---
        tensor1 = new Tsr(new int[]{3, 2, 1}, 4);
        tensor2 = new Tsr(new int[]{1, 1, 4}, -1);
        tensor3 = new Tsr(new int[]{3, 2, 1}, 2);
        tensor2.setRqsGradient(true);
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1, tensor2, tensor3},
                "I[0]xI[1]xI[2]",
                new String[]{
                        "[1x1x4]:(-48.0, -48.0, -48.0, -48.0);",
                        " =>d|[ [3x2x1]:(2.0, 2.0, 2.0, 2.0, 2.0, 2.0) ]|" +
                                ":t{ [3x2x4]:(-4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0); ",
                        "=>d|[ [3x2x1]:(4.0, 4.0, 4.0, 4.0, 4.0, 4.0) ]|" +
                                ":t{ [1x1x4]:(-1.0, -1.0, -1.0, -1.0) },  }, "
                });
        //--
        tensor1 = new Tsr(new int[]{5, 1, 1}, 4);//-2*4 = 8 | *3 = -24
        tensor2 = new Tsr(new int[]{1, 4, 1}, -2);
        tensor3 = new Tsr(new int[]{1, 1, 2}, 3);
        tensor1.setRqsGradient(true);
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1, tensor2, tensor3},
                "I[0]xI[1]xI[2]",
                new String[]{
                        "[5x4x2]:(-24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0); =>d|[ [1x1x2]:(3.0, 3.0) ]|:t{ [5x4x1]:(-8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0); =>d|[ [1x4x1]:(-2.0, -2.0, -2.0, -2.0) ]|:t{ [5x1x1]:(4.0, 4.0, 4.0, 4.0, 4.0) },  }, "
                }
        );
        //=====================
        tensor1 = new Tsr(new int[]{2, 2}, new double[]{1, 2, 3, 4});//-2*4 = 8 | *3 = -24
        tensor1.setRqsGradient(true);
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1},
                "((i0+2)^2)",
                new String[]{"d|[ [2x2]:(6.0, 8.0, 10.0, 12.0) ]|:t{ [2x2]:(1.0, 2.0, 3.0, 4.0) }"}
        );
        //---
        //=====================
        tensor1 = new Tsr(new int[]{2}, new double[]{1, 2});//-2*4 = 8 | *3 = -24
        tensor1.setRqsGradient(true);
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1},
                "cos(tanh(lig(i0)))",
                new String[]{"[2]:(0.6998554841989726, 0.6177112361351595); =>d|[ [2]:(-0.1916512536291578, -0.12539562877521598) ]|:t{ [2]:(1.0, 2.0) }, "}
        );
        //---
        //=====================
        tensor1 = new Tsr(new int[]{1}, new double[]{2});//-2*4 = 8 | *3 = -24
        tensor1.setRqsGradient(true);
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1},//2 =>-2 =>-4 =>12 //2 =>-2 //-2,12 =>-24
                "(-3*(2*(i0*-1)))*(-1*i0)",
                new String[]{
                        "[1]:(-24.0); ",
                        "=>d|[ [1]:(12.0) ]|:" +
                                "t{ [1]:(-2.0); " +
                                "=>d|[ [1]:(-1.0) ]|:" +
                                "t{ [1]:(2.0) },  " +
                                "},",
                        "=>d|[ [1]:(-2.0) ]|:" +
                                "t{ [1]:(12.0); " +
                                "=>d|[ [1]:(6.0) ]|:" +
                                "t{ [1]:(2.0) },  " +
                                "},"
                }
        );
        //---
        //======================
        //TESTING INVERS:
        ///=================
        //---
        tensor1 = new Tsr(new int[]{2, 3}, new double[]{
                0, 0, //3, 1,
                0, 0, //-2, -1,
                0, 0, ///-4, 2,
        });
        //---
        tensor2 = new Tsr(new int[]{5, 2}, new double[]{
                -2, 3, 6, 3, -1,
                0, 2, 4, 2, 1
        });
        tensor3 = new Tsr(new int[]{4, 2}, new double[]{
                1, 2, -3, 2,//<= drain data!
                4, -2, -1, 5,
        });
        tester.testTensorAutoGrad(new Tsr[]{tensor1, tensor2, tensor3},
                "i0<<i1<<i2",
                new String[]{"empty"});
        result = IFunction.setup.commit(new Tsr[]{tensor1, tensor2, tensor3}, "i0<<i1<<i2", true);
        tester.testContains(
                result.toString(),
                new String[]{"[2x3]:(-8.0, 4.0, -9.0, -2.0, 2.0, 3.0)"},
                "Testing if IFunction.setup.commit() returns non unique result!"
        );
        tester.testTensorAutoGrad(new Tsr[]{tensor1, tensor2, tensor3},//TODO:REACTIVATE!
                "i2>>i1>>i0",
                new String[]{"empty"});
        result = IFunction.setup.commit(new Tsr[]{tensor1, tensor2, tensor3}, "i2>>i1>>i0", true);
        tester.testContains(
                result.toString(),
                new String[]{"[2x3]:(-8.0, 4.0, -9.0, -2.0, 2.0, 3.0)"},
                "Testing if IFunction.setup.commit() returns non unique result!"
        );
        //=====================
        //---
        tensor1 = new Tsr(new int[]{2, 2, 1}, new double[]{
                1, 2, //3, 1,
                2, -3, //-2, -1,
        });
        tensor1.setRqsGradient(true);
        //---
        tensor2 = new Tsr(new int[]{1, 2, 2}, new double[]{
                -2, 3,
                1, 2,
        });
        tester.testTensorAutoGrad(//4, 5, -13, -4 <= result values
                new Tsr[]{tensor1, tensor2},
                "i0xi1",
                new String[]{"[2x1x2]:(4.0, -13.0, 5.0, -4.0); =>d|[ [1x2x2]:(-2.0, 3.0, 1.0, 2.0) ]|:t{ [2x2x1]:(1.0, 2.0, 2.0, -3.0) }"},
                new Tsr(new int[]{2, 1, 2}, new double[]{1, 1, 1, 1}),
                new double[][]{{-1.0, -1.0, 5.0, 5.0}, null}
        );
        //---
        //======================
        int[] shape = {4, 2, 9, 5, 6, 2};
        int[] newForm = {1, 0, -1, -2, 2, 4, 3, -1, 5};
        int[] expected = {2, 4, 1, 2, 9, 6, 5, 1, 2};
        tester.testTensorUtility_reshape(shape, newForm, expected);
        //---
        shape = new int[]{4, 2, 9, 5, 6, 2};
        expected = new int[]{1, 4, 4 * 2, 4 * 2 * 9, 4 * 2 * 9 * 5, 4 * 2 * 9 * 5 * 6};
        tester.testTensorUtility_translation(shape, expected);
        //---
        shape = new int[]{4, 2, 9, 5, 6, 2};
        expected = new int[]{1, 1, 4, 0, 0, 0};
        tester.testTensorBase_idxFromAnchor(shape, 37, expected);
        //---
        shape = new int[]{4, 3, 2, 5};
        expected = new int[]{3, 2, 1, 2};
        int idx = (1) * 3 + (1 * 4) * 2 + (1 * 4 * 3) * 1 + (1 * 4 * 3 * 2) * 2;
        tester.testTensorBase_idxFromAnchor(shape, idx, expected);
        //---
        int[] frstShape = {4, 1};
        double[] frstData = {
                -2, -1, 4, 3,
        };
        int[] scndShape = {1, 3};
        double[] scndData = {
                2,
                -4,
                3,
        };
        tester.testTensCon(
                frstShape, scndShape,
                frstData, scndData,
                new double[]{
                        -4, -2, 8, 6,
                        8, 4, -16, -12,
                        -6, -3, 12, 9
                }
        );
        //---
        frstShape = new int[]{4, 2};
        frstData = new double[]{
                -1, 2, -3, 1,
                4, -5, 2, 3,
        };
        scndShape = new int[]{2, 3};
        scndData = new double[]{
                4, -2,
                8, -1,
                -5, 2,
        };
        tester.testTensCon(
                frstShape, scndShape,
                frstData, scndData,
                new double[]{
                        29, -28, -1,
                        -40, 48, -29,
                }
        );
        //---
        frstShape = new int[]{1, 1};
        frstData = new double[]{2};

        scndShape = new int[]{2, 3};
        scndData = new double[]{
                4, -2,
                8, -1,
                -5, 2,
        };
        tester.testTensCon(
                frstShape, scndShape,
                frstData, scndData,
                new double[]{
                        8, -4,
                        16, -2,
                        -10, 4,
                }
        );
        //---
        frstShape = new int[]{2, 3, 2};
        frstData = new double[]{
                1, 2,
                3, 4,
                0, 2,

                3, 4,
                2, -1,
                -2, -3
        };
        //---
        scndShape = new int[]{2, 2, 3};
        scndData = new double[]{
                -1, 3,
                0, 2,

                -3, 1,
                2, -3,

                0, 4,
                5, -1
        };
        tester.testTensCon(
                frstShape, scndShape,
                frstData, scndData,
                new double[]{
                        15, 11,
                        20, -22,
                }
        );
        //---
        //TESTING INVERS:
        ///=================
        //---
        frstShape = new int[]{2, 3};
        frstData = new double[]{
                0, 0, //3, 1,
                0, 0, //-2, -1,
                0, 0, ///-4, 2,
        };
        //---
        scndShape = new int[]{5, 2};
        scndData = new double[]{
                -2, 3, 6, 3, -1,
                0, 2, 4, 2, 1
        };
        tester.testInvTensCon(//
                frstShape, scndShape,
                frstData, scndData,
                new double[]{
                        1, 2, -3, 2,//<= drain data!
                        4, -2, -1, 5,
                },
                new double[]{
                        1 * -2 + 2 * 3 - 3 * 6 + 2 * 3, 1 * 3 + 2 * 6 - 3 * 3 + 2 * -1,
                        1 * 0 + 2 * 2 - 3 * 4 + 2 * 2 + 4 * -2 - 2 * 3 - 1 * 6 + 5 * 3, 1 * 2 + 2 * 4 - 3 * 2 + 2 * 1 + 4 * 3 - 2 * 6 - 1 * 3 + 5 * -1,
                        4 * 0 - 2 * 2 - 1 * 4 + 5 * 2, 4 * 2 - 2 * 4 - 1 * 2 + 5 * 1
                },
                true
        );
        //===========================================

        tensor1 = new Tsr(new int[]{3, 5}, new double[]{
                2, 3, 5,
                -4, 6, 2,
                -5, -2, -1,
                2, 4, -1,
                1, 2, 7
        });
        if(!System.getProperty("os.name").toLowerCase().contains("windows")){
            return;
        }
        //=====================================================================
        Device gpu = new Device("nvidia");
        gpu.add(tensor1);
        //System.out.println(new Tsr(t, "lig(I[0])"));
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1},
                "lig(I[0])",
                new String[]{
                        "[3x5]:(2.1269280110429722, 3.048587351573742, 5.006715348489118, 0.01814992791780978, 6.00247568513773, 2.1269280110429722, 0.006715348489117967, 0.1269280110429726, 0.31326168751822286, 2.1269280110429722, 4.0181499279178094, 0.31326168751822286, 1.3132616875182228, 2.1269280110429722, 7.000911466453774)"
                });
        //===================
        tensor1 = new Tsr(new int[]{2}, 3);
        tensor2 = new Tsr(new int[]{2}, 4);
        result = new Tsr(new Tsr[]{tensor1, tensor2}, "i0*i1");
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1, tensor2},
                "i0*i1",
                new String[]{"[2]:(12.0, 12.0)"});
        //===========================================
        tensor1 = new Tsr(
                new int[]{2, 3, 1},
                new double[]{
                        3, 2,
                        -1, -2,
                        2, 4
                }
        );
        tensor2 = new Tsr(
                new int[]{1, 3, 2},
                new double[]{
                        4, -1, 3,
                        2, 3, -1
                });
        gpu.add(tensor1);
        gpu.add(tensor2);
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1, tensor2},
                "I0 x i1",
                new String[]{
                        "[2x1x2]:(19.0, 22.0, 1.0, -6.0)"
                });
        //=======================
        tensor1 = new Tsr(
                new int[]{200, 300, 1},
                2
        );
        tensor2 = new Tsr(
                new int[]{1, 300, 200},
                3);
        gpu.add(tensor1);
        gpu.add(tensor2);
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1, tensor2},
                "I0xi1",
                new String[]{
                        "[200x1x200]:(1800.0, 1800.0, 1800.0, 1800.0, 1800.0, 1800.0,"//...
                });
        //---
        tensor1 = new Tsr(new int[]{2, 2, 1}, new double[]{
                1, 2, //3, 1,
                2, -3, //-2, -1,
        }).setRqsGradient(true);
        //---
        tensor2 = new Tsr(new int[]{1, 2, 2}, new double[]{
                -2, 3,
                1, 2,
        });
        gpu.add(tensor1).add(tensor2);
        tester.testTensorAutoGrad(//4, 5, -13, -4 <= result values
                new Tsr[]{tensor1, tensor2},
                "i0xi1",
                new String[]{
                        "[2x1x2]:(4.0, -13.0, 5.0, -4.0); =>d|[ [1x2x2]:(-2.0, 3.0, 1.0, 2.0) ]|:t{ [2x2x1]:(1.0, 2.0, 2.0, -3.0) }"
                },
                new Tsr(new int[]{2, 1, 2}, new double[]{1, 1, 1, 1}),
                new double[][]{{-1.0, -1.0, 5.0, 5.0}, null}
        );
        //result = new Tsr(new Tsr[]{tensor1, tensor1}, "ig0<-i0");
        //tester.testContains(tensor1.toString("g"), new String[]{"test"}, "");
        Tsr x = new Tsr(new int[]{1}, 3).setRqsGradient(true);
        Tsr b = new Tsr(new int[]{1}, -4);
        Tsr w = new Tsr(new int[]{1}, 2);
        gpu.add(x).add(b).add(w);
        /**
         *      ((3-4)*2)^2 = 4
         *  dx:   8*3 - 32  = -8
         * */
        Tsr y = new Tsr(new Tsr[]{x, b, w}, "((i0+i1)*i2)^2");
        tester.testTensor(y, new String[]{"[1]:(4.0); ->d[1]:(-8.0), "});
        y.backward(new Tsr(2));
        tester.testTensor(x, new String[]{"-16.0"});
        tester.testShareDevice(gpu, new Tsr[]{y, x, b, w});
        //---
        x = new Tsr(new int[]{1}, 4).setRqsGradient(true);
        b = new Tsr(new int[]{1}, 0.5);
        w = new Tsr(new int[]{1}, 0.5);
        y = new Tsr(new Tsr[]{x, b, w}, "(2^i0^i1^i2^2");//TODO: fix!
        //tester.testTensor(y, new String[]{"[1]:(4.0);", " ->d[1]:(-8.0), "});
        //tester.testShareDevice(gpu, new Tsr[]{y, x, b, w});

        //====
        x = new Tsr(new int[]{1}, 3);
        b = new Tsr(new int[]{1}, -5);
        w = new Tsr(new int[]{1}, -2);
        gpu.add(x).add(b).add(w);
        Tsr z = new Tsr(new Tsr[]{x, b, w}, "I0*i1*i2");
        tester.testTensor(z, new String[]{"[1]:(30.0)"});
        tester.testShareDevice(gpu, new Tsr[]{z, x, b, w});
        //---
        try {
            Thread.sleep(10000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        tester.closeWindows();
    }

    @Test
    public void testTensorDevice() {
        if(!System.getProperty("os.name").toLowerCase().contains("windows")){
            return;
        }
        NTester_TensorDevice tester = new NTester_TensorDevice("Testing tensor device");
        Device gpu = new Device("nvidia");
        Tsr tensor = Tsr.factory.newTensor(new double[]{1, 3, 4, 2, -3, 2, -1, 6}, new int[]{2, 4});
        Tsr firstTensor = tensor;
        tester.testAddTensor(gpu, tensor,
                new double[]{1, 3, 4, 2, -3, 2, -1, 6},
                new int[]{2, 4},
                new int[]{1, 2},
                new int[]{0, 8, 0, 2, 0, 2});
        tensor = Tsr.factory.newTensor(new double[]{-7, -9}, new int[]{2});
        tester.testAddTensor(gpu, tensor,
                new double[]{1, 3, 4, 2, -3, 2, -1, 6, -7, -9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,},
                new int[]{2, 4},
                new int[]{1, 2},
                new int[]{0, 8, 0, 2, 0, 2, 8, 2, 0, 1, 0, 1});
        tensor = Tsr.factory.newTensor(new double[]{4, -4, 9, 4, 77}, new int[]{5});
        tester.testAddTensor(gpu, tensor,
                new double[]{1, 3, 4, 2, -3, 2, -1, 6, -7, -9, 4, -4, 9, 4, 77, 0, 0, 0, 0, 0,},
                new int[]{2, 4, 5, 0},
                new int[]{1, 2},
                new int[]{0, 8, 0, 2, 0, 2, 8, 2, 0, 1, 0, 1, 10, 5, 2, 1, 0, 1});
        tester.testGetTensor(gpu, firstTensor,
                new double[]{1, 3, 4, 2, -3, 2, -1, 6, -7, -9, 4, -4, 9, 4, 77, 0, 0, 0, 0, 0,},
                new int[]{2, 4, 5, 0},
                new int[]{1, 2},
                new int[]{8, 2, 0, 1, 0, 1, 10, 5, 2, 1, 0, 1}
        );
        tester.testAddTensor(gpu, firstTensor,
                new double[]{1, 3, 4, 2, -3, 2, -1, 6, -7, -9, 4, -4, 9, 4, 77, 0, 0, 0, 0, 0,},
                new int[]{2, 4, 5, 0},
                new int[]{1, 2},
                new int[]{0, 8, 0, 2, 0, 2, 8, 2, 0, 1, 0, 1, 10, 5, 2, 1, 0, 1}
        );
        tester.testGetTensor(gpu, firstTensor,
                new double[]{1, 3, 4, 2, -3, 2, -1, 6, -7, -9, 4, -4, 9, 4, 77, 0, 0, 0, 0, 0,},
                new int[]{2, 4, 5, 0},
                new int[]{1, 2},
                new int[]{8, 2, 0, 1, 0, 1, 10, 5, 2, 1, 0, 1}
        );
        tensor = Tsr.factory.newTensor(new double[]{888, 777, -33, 999}, new int[]{2, 2});
        tensor.setRqsGradient(true);
        tester.testAddTensor(gpu, tensor,
                new double[]{888, 777, -33, 999, 0, 0, 0, 0, -7, -9, 4, -4, 9, 4, 77, 0, 0, 0, 0, 0,},
                new int[]{2, 4, 5, 2, 2, 0, 0, 0},
                new int[]{1, 2},
                new int[]{0, -4, 3, 2, 0, 2, 8, 2, 0, 1, 0, 1, 10, 5, 2, 1, 0, 1}
        );
        //TESTING TENS MUL ON GPU NOW!!!!!
        Tsr src1 = new Tsr(
                new int[]{2, 3, 2},
                new double[]{
                        1, 2,
                        3, 4,
                        0, 2,

                        3, 4,
                        2, -1,
                        -2, -3
                }
        );
        Tsr src2 = new Tsr(
                new int[]{2, 2, 3},
                new double[]{
                        -1, 3,
                        0, 2,

                        -3, 1,
                        2, -3,

                        0, 4,
                        5, -1
                }
        );
        Tsr drn = new Tsr(
                new int[]{1, 2, 2},
                new double[]{
                        0, 0,
                        0, 0,
                }
        );
        gpu.add(src1);
        gpu.add(src2);
        gpu.add(drn);
        tester.testCalculation(
                gpu, drn, src1, src2, 18, -1,//Tsr mul
                new double[]{888.0, 777.0, -33.0, 999.0, 0.0, 0.0, 0.0, 0.0, -7.0, -9.0, 4.0, -4.0, 9.0, 4.0, 77.0, 1.0, 2.0, 3.0, 4.0, 0.0, 2.0, 3.0, 4.0, 2.0, -1.0, -2.0, -3.0, -1.0, 3.0, 0.0, 2.0, -3.0, 1.0, 2.0, -3.0, 0.0, 4.0, 5.0, -1.0, 15.0, 11.0, 20.0, -22.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
        );
        tester.testCalculation(
                gpu, drn, src1, src2, 18, -1,//Tsr mul
                new double[]{888.0, 777.0, -33.0, 999.0, 0.0, 0.0, 0.0, 0.0, -7.0, -9.0, 4.0, -4.0, 9.0, 4.0, 77.0, 1.0, 2.0, 3.0, 4.0, 0.0, 2.0, 3.0, 4.0, 2.0, -1.0, -2.0, -3.0, -1.0, 3.0, 0.0, 2.0, -3.0, 1.0, 2.0, -3.0, 0.0, 4.0, 5.0, -1.0, 15.0, 11.0, 20.0, -22.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
        );
        //INVERS OF CONV
        Tsr d_src1 = new Tsr(new int[]{2, 3}, new double[]{
                0, 0, //3, 1,
                0, 0, //-2, -1,
                0, 0, ///-4, 2,
        });
        Tsr d_src2 = new Tsr(new int[]{5, 2}, new double[]{
                -2, 3, 6, 3, -1,
                0, 2, 4, 2, 1
        });
        Tsr d_drn = new Tsr(new int[]{4, 2}, new double[]{
                1, 2, -3, 2,
                4, -2, -1, 5,
        });
        double[] expectedInv = new double[]{
                1 * -2 + 2 * 3 - 3 * 6 + 2 * 3, 1 * 3 + 2 * 6 - 3 * 3 + 2 * -1,
                1 * 0 + 2 * 2 - 3 * 4 + 2 * 2 + 4 * -2 - 2 * 3 - 1 * 6 + 5 * 3, 1 * 2 + 2 * 4 - 3 * 2 + 2 * 1 + 4 * 3 - 2 * 6 - 1 * 3 + 5 * -1,
                4 * 0 - 2 * 2 - 1 * 4 + 5 * 2, 4 * 2 - 2 * 4 - 1 * 2 + 5 * 1
        };
        gpu.add(d_src1).add(d_src2).add(d_drn);
        gpu.printDeviceContent(true);
        tester.testCalculation(gpu, d_drn, d_src1, d_src2, 18, 0, expectedInv);
        //ADDITION
        System.out.println("WE ARE HERE:");
        gpu.printDeviceContent(true);
        //addition:
        drn = new Tsr(
                new int[]{3, 2, 2},
                new double[]{
                        111, 222, 333,
                        444, 555, 666,
                        777, 888, 999,
                        1098, 32, 150,
                }
        );
        //Adding new drain:
        tester.testAddTensor(gpu, drn,
                new double[]{888.0, 777.0, -33.0, 999.0, 0.0, 0.0, 0.0, 0.0, -7.0, -9.0, 4.0, -4.0, 9.0, 4.0, 77.0, 1.0, 2.0, 3.0, 4.0, 0.0, 2.0, 3.0, 4.0, 2.0, -1.0, -2.0, -3.0, -1.0, 3.0, 0.0, 2.0, -3.0, 1.0, 2.0, -3.0, 0.0, 4.0, 5.0, -1.0, 15.0, 11.0, 20.0, -22.0, -8.0, 4.0, -9.0, -2.0, 2.0, 3.0, -2.0, 3.0, 6.0, 3.0, -1.0, 0.0, 2.0, 4.0, 2.0, 1.0, 1.0, 2.0, -3.0, 2.0, 4.0, -2.0, -1.0, 5.0, 111.0, 222.0, 333.0, 444.0, 555.0, 666.0, 777.0, 888.0, 999.0, 1098.0, 32.0, 150.0,},
                new int[]{2, 4, 5, 2, 2, 3, 2, 1, 2, 2, 4, 2, 3, 2, 2,},
                new int[]{1, 2, 1, 2, 6, 1, 2, 4, 1, 1, 2, 1, 5, 1, 4, 1, 3, 6,},
                new int[]{0, -4, 3, 2, 0, 2, 8, 2, 0, 1, 0, 1, 10, 5, 2, 1, 0, 1, 15, 12, 4, 3, 2, 3, 27, 12, 3, 3, 5, 3, 39, 4, 7, 3, 8, 3, 43, 6, 4, 2, 0, 2, 49, 10, 2, 2, 11, 2, 59, 8, 10, 2, 13, 2, 67, 12, 12, 3, 15, 3,}
        );
        tester.testCalculation(
                gpu,
                drn, src1, src2, 17, -1,//Tsr overwrite
                new double[]{888.0, 777.0, -33.0, 999.0, 0.0, 0.0, 0.0, 0.0, -7.0, -9.0, 4.0, -4.0, 9.0, 4.0, 77.0, 1.0, 2.0, 3.0, 4.0, 0.0, 2.0, 3.0, 4.0, 2.0, -1.0, -2.0, -3.0, -1.0, 3.0, 0.0, 2.0, -3.0, 1.0, 2.0, -3.0, 0.0, 4.0, 5.0, -1.0, 15.0, 11.0, 20.0, -22.0, -8.0, 4.0, -9.0, -2.0, 2.0, 3.0, -2.0, 3.0, 6.0, 3.0, -1.0, 0.0, 2.0, 4.0, 2.0, 1.0, 1.0, 2.0, -3.0, 2.0, 4.0, -2.0, -1.0, 5.0, 0.0, 5.0, 3.0, 6.0, -3.0, 3.0, 5.0, 1.0, 2.0, 3.0, 3.0, -4.0,}
        );
        tester.testCalculation(
                gpu,
                drn, drn, null, 6, -1,//Tsr gaus
                new double[]{888.0, 777.0, -33.0, 999.0, 0.0, 0.0, 0.0, 0.0, -7.0, -9.0, 4.0, -4.0, 9.0, 4.0, 77.0, 1.0, 2.0, 3.0, 4.0, 0.0, 2.0, 3.0, 4.0, 2.0, -1.0, -2.0, -3.0, -1.0, 3.0, 0.0, 2.0, -3.0, 1.0, 2.0, -3.0, 0.0, 4.0, 5.0, -1.0, 15.0, 11.0, 20.0, -22.0, -8.0, 4.0, -9.0, -2.0, 2.0, 3.0, -2.0, 3.0, 6.0, 3.0, -1.0, 0.0, 2.0, 4.0, 2.0, 1.0, 1.0, 2.0, -3.0, 2.0, 4.0, -2.0, -1.0, 5.0, 1.0, 1.3887943864964039E-11, 1.2340980408667962E-4, 2.3195228302435736E-16, 1.2340980408667962E-4, 1.2340980408667962E-4, 1.3887943864964039E-11, 0.36787944117144233, 0.018315638888734182, 1.2340980408667962E-4, 1.2340980408667962E-4, 1.1253517471925921E-7,}
        );
        //---
        gpu.getKernel().dispose();
        System.out.println("Done!");
        try {
            Thread.sleep(2000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        tester.closeWindows();
    }


}