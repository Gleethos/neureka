import neureka.Neureka;
import neureka.Tsr;
import neureka.calculus.Function;
import neureka.calculus.factory.assembly.FunctionBuilder;
import neureka.autograd.GraphNode;
import org.junit.Test;
import util.NTester_Tensor;

public class BroadJavaTests {

    @Test
    public void test_autograd()
    {
        Neureka.instance().reset();
        Neureka.instance().settings().autoDiff().setApplyGradientWhenTensorIsUsed(false);
        Neureka.instance().settings().view().setLegacy(true);

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
        tester.testTensor(x, new String[]{"-24.0"});

        y = new Tsr("((",x,"+",b,")*",w,")^2");
        tester.testTensor(y, new String[]{"[1]:(4.0); ->d[1]:(-8.0), "});
        y.backward(new Tsr(-1));
        tester.testTensor(x, new String[]{"-16.0"});
        //===========================================
        Neureka.instance().settings().indexing().setLegacy(true);
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

        Neureka.instance().settings().indexing().setLegacy(false);
        //Same test again but this time with reversed indexing:
        x = new Tsr(
                new int[]{2, 3, 1},
                new double[]{
                        3, 2, -1,
                        -2, 2, 4
                }
        );
        y = new Tsr(
                new int[]{1, 3, 2},
                new double[]{
                        4, -1,
                        3,  2,
                        3, -1
                });
        /*
                15, 2,
                10, 2
         */
        z = new Tsr(new Tsr[]{x, y}, "I0xi1");
        tester.testTensor(z, new String[]{"[2x1x2]:(15.0, 2.0, 10.0, 2.0)"});
        z = new Tsr(new Object[]{x, "x", y});
        tester.testTensor(z, new String[]{"[2x1x2]:(15.0, 2.0, 10.0, 2.0)"});
        //=======================
        Neureka.instance().settings().indexing().setLegacy(true);
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
        //---
        Neureka.instance().settings().indexing().setLegacy(false);
        //--- again but now reverse: (outcome should not change...)
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
        //TODO: add tests using more then 1 function and check if the graph is build correctly!
    }

    @Test
    public void test_tensor_operations_and_autograd() {
        Neureka.instance().reset();
        Neureka.instance().settings().indexing().setLegacy(true);
        Neureka.instance().settings().view().setLegacy(true);

        NTester_Tensor tester = new NTester_Tensor("Testing core tensor functionality");

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
        Function tanh = FunctionBuilder.build("tanh(i0)", true);
        Function tenxx = FunctionBuilder.build("i0*100", true);
        z = tenxx.call(new Tsr[]{tanh.call(new Tsr[]{x})});
        tester.testTensor(z, new String[]{"[1]:(9.95037E0)"});
        Neureka.instance().settings().debug().setKeepDerivativeTargetPayloads(true);
        z.backward(new Tsr(new int[]{1}, 1));
        tester.testTensor(x, new String[]{"[1]:(0.1):g:(99.0099E0)"});
        tester.testTensor(z, new String[]{"[1]:(9.95037E0); ->d[1]:(99.0099E0), "});
        Neureka.instance().settings().debug().setKeepDerivativeTargetPayloads(false);
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
        //---
        Tsr tensor1 = new Tsr(3).setRqsGradient(true);
        Tsr tensor2 = new Tsr(-4);
        Tsr tensor3 = new Tsr(2);
        tester.testInjection(
                new Tsr[]{tensor1, tensor2, tensor3},//ERROR here
                "(Ig[0]<-I[1])->I[2]",
                new String[][]{
                        {"empty"},//result ... Why? : Identity must not be lost ! (new Tsr() cannot be member of inputs...)
                        {"(3.0)", "g:(-4.0)"},//tensor1
                        {"(-4.0)"},//tensor2
                        {"(-4.0)"},//tensor3
                });
        tensor1 = new Tsr(3).setRqsGradient(true);
        tensor2 = new Tsr(-4);
        tensor3 = new Tsr(2);
        Tsr result = Function.Setup.commit(new Tsr[]{tensor1, tensor2, tensor3}, "(Ig[0]<-I[1])->I[2]", true);
        tester.testContains(
                result.toString(),
                new String[]{"(-4.0)"},
                "Testing if Function.setup.commit() returns non unique result!"
        );
        //---
        tensor1 = new Tsr(new int[]{1, 3}, 2);
        tensor2 = new Tsr(new int[]{2, 1}, -1);
        tensor1.setRqsGradient(true);
        tensor2.setRqsGradient(true);
        Neureka.instance().settings().debug().setKeepDerivativeTargetPayloads(true);
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
        //---//Broken down to 2 functions:
        tensor1 = new Tsr(new int[]{1, 3}, 2);
        tensor2 = new Tsr(new int[]{2, 1}, -1);
        tensor1.setRqsGradient(true);
        tensor2.setRqsGradient(true);
        result = new Tsr(new Tsr[]{tensor1, tensor2}, "I[0]xI[1]");
        result = new Tsr(new Tsr[]{result}, "lig(I[0]*-100)");
        tester.testContains(result.toString("rc"),
                new String[]{
                        "[2x3]:(200.0, 200.0, 200.0, 200.0, 200.0, 200.0);",
                        " =>d|[ [2x3]:(-100.0, -100.0, -100.0, -100.0, -100.0, -100.0) ]|:t{ [2x3]:(-2.0, -2.0, -2.0, -2.0, -2.0, -2.0);",
                        " =>d|[ [1x3]:(2.0, 2.0, 2.0) ]|:t{ [2x1]:(-1.0, -1.0) },",
                        " =>d|[ [2x1]:(-1.0, -1.0) ]|:t{ [1x3]:(2.0, 2.0, 2.0) },",
                        "  }, "
                }, "");
        Neureka.instance().settings().debug().setKeepDerivativeTargetPayloads(false);
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
                        "[2]:(4.0105E0, 4.0105E0); ",
                        "=>d|[ [2]:(3.9275E0, 3.9275E0) ]|" +
                                ":t{ [2]:(0.99805E0, 0.99805E0); ",
                                        "=>d|[ [2]:(0.01556E0, 0.01556E0) ]|" +
                                                ":t{ [2]:(4.0, 4.0) }, ",
                                        "=>d|[ [2]:(0.03112E0, 0.03112E0) ]|" +
                                                ":t{ [2]:(2.0, 2.0) }, ",
                                "}, ",
                        "=>d|[ [2]:(0.97996E0, 0.97996E0) ]|" +
                                ":t{ [2]:(4.0, 4.0) }, "
                }
        );
        //---
        Neureka.instance().settings().debug().setKeepDerivativeTargetPayloads(true);
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
                                        ":t{ [1x1x4]:(-1.0, -1.0, -1.0, -1.0) },  " +
                                "}, "
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
        Neureka.instance().settings().debug().setKeepDerivativeTargetPayloads(false);
        //=====================
        tensor1 = new Tsr(new int[]{2, 2}, new double[]{1, 2, 3, 4});//-2*4 = 8 | *3 = -24
        tensor1.setRqsGradient(true);
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1},
                "((i0+2)^2)",
                new String[]{"d|[ [2x2]:(6.0, 8.0, 10.0, 12.0) ]|:t{ [2x2]:(1.0, 2.0, 3.0, 4.0) }"}
        );
        //---
        tensor1 = new Tsr(new int[]{2, 2}, new double[]{1, 2, 3, 4});//-2*4 = 8 | *3 = -24
        tensor1.setRqsGradient(true);
        result = new Tsr(new Tsr[]{tensor1}, "(i0+2)");
        result = new Tsr(new Tsr[]{result}, "I[0]^2");
        tester.testContains(
                result.toString("rc"),
                new String[]{"d|[ [2x2]:(6.0, 8.0, 10.0, 12.0) ]|:t{ [2x2]:(1.0, 2.0, 3.0, 4.0) }"},
                ""
        );
        //=====================
        tensor1 = new Tsr(new int[]{2}, new double[]{1, 2});//-2*4 = 8 | *3 = -24
        tensor1.setRqsGradient(true);
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1},
                "cos(tanh(lig(i0)))",
                new String[]{"[2]:(0.69985E0, 0.61771E0); =>d|[ [2]:(-0.19165E0, -0.12539E0) ]|:t{ [2]:(1.0, 2.0) }, "}
        );
        //---
        //=====================
        tensor1 = new Tsr(-3).setRqsGradient(true);
        tensor2 = new Tsr(4).setRqsGradient(true);
        tensor3 = new Tsr(2);
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1, tensor2, tensor3},
                "(relu(i0*i1)+i1)/i2",
                new String[]{
                    "[1]:(1.94);",
                        "=>d|[ [1]:(0.5) ]|:",
                            "t{",
                                "[1]:(-0.12);",
                                    "=>d|[ [1]:(-0.03) ]|:",
                                        "t{ [1]:(4.0) },",
                                    "=>d|[ [1]:(0.04) ]|:",
                                        "t{ [1]:(-3.0) },",
                            "},",
                        "=>d|[ [1]:(0.5) ]|:",
                            "t{",
                                "[1]:(4.0)",
                            "},"}
        );
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
        result = new Tsr(new Tsr[]{tensor1}, "(-3*(2*(i0*-1)))*(-1*i0)");
        GraphNode node = (GraphNode) result.find(GraphNode.class);
        String asString = node.toString("g");
        tester.testContains(
                asString,
                new String[]{
                        "[1]:(-24.0)",
                        "[1]:(12.0)",
                        "[1]:(2.0)",
                        "f(NONE) => [1]:(-3.0)",
                        "(-1.0*I[0])",
                        "(I[0]*-1.0)",
                        "f(I[0]*I[1])",
                        "LEAVE RQS GRADIENT"
                },
                "Testing 'toString' of GraphNode");
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
                "i0<<xi1<<xi2",
                new String[]{"empty"});
        result = Function.Setup.commit(new Tsr[]{tensor1, tensor2, tensor3}, "i0<<xi1<<xi2", true);
        tester.testContains(
                result.toString(),
                new String[]{"[2x3]:(-8.0, 4.0, -9.0, -2.0, 2.0, 3.0)"},
                "Testing if Function.setup.commit() returns non unique result!"
        );
        tester.testTensorAutoGrad(new Tsr[]{tensor1, tensor2, tensor3},//TODO:REACTIVATE!
                "i2x>>i1x>>i0",
                new String[]{"empty"});
        result = Function.Setup.commit(new Tsr[]{tensor1, tensor2, tensor3}, "i2x>>i1x>>i0", true);
        tester.testContains(
                result.toString(),
                new String[]{"[2x3]:(-8.0, 4.0, -9.0, -2.0, 2.0, 3.0)"},
                "Testing if Function.setup.commit() returns non unique result!"
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
        Neureka.instance().settings().debug().setKeepDerivativeTargetPayloads(true);
        tester.testTensorAutoGrad(//4, 5, -13, -4 <= result values
                new Tsr[]{tensor1, tensor2},
                "i0xi1",
                new String[]{"[2x1x2]:(4.0, -13.0, 5.0, -4.0); =>d|[ [1x2x2]:(-2.0, 3.0, 1.0, 2.0) ]|:t{ [2x2x1]:(1.0, 2.0, 2.0, -3.0) }"},
                new Tsr(new int[]{2, 1, 2}, new double[]{1, 1, 1, 1}),
                new double[][]{{-1.0, -1.0, 5.0, 5.0}, new double[0]}
        );
        Neureka.instance().settings().debug().setKeepDerivativeTargetPayloads(false);
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

        tester.testTensorUtility_makeFit(
                new int[]{1, 7, 9, 2, 1, 1},
                new int[]{1, 1, 3, 2, 7, 5, 1},
                new int[][]{
                        new int[]{ 0,  1,  2, 3, -1, -1, -1, -1},
                        new int[]{-1, -1, -1, 2,  3,  4,  5,  6}
                }
        );

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
        //---

        //---
        frstShape = new int[]{1, 3};
        frstData = new double[]{
                2,
                -1,
                3
        };

        scndShape = new int[]{2, 1};
        scndData = new double[]{4, -2,};
        tester.testTensBroadcast(
                frstShape, scndShape,
                frstData, scndData,
                new double[]{
                        8, -4,
                        -4, 2,
                        12, -6,
                }
        );
        //---
        //---
        //---invers:
        frstShape = new int[]{1, 3};
        frstData = new double[]{
                0,
                0,
                0
        };
        //---
        scndShape = new int[]{2, 1};
        scndData = new double[]{
                2, -1,
        };
        tester.testInvTensBroadcast(//
                frstShape, scndShape,
                frstData, scndData,
                new double[]{
                        8, -4,
                        -4, 2,
                        12, -6,
                },
                new double[]{
                     16+4,//20
                     -8-2,//-10
                     24+6//30
                },
                true
        );
        //===========================================
    }


}
