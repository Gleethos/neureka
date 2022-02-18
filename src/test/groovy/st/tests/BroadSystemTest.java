package st.tests;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.GraphNode;
import neureka.calculus.Function;
import neureka.calculus.assembly.FunctionBuilder;
import testutility.UnitTester_Tensor;

public class BroadSystemTest
{

    public static boolean on()
    {
        Neureka.get().settings().view().getTensorSettings().setIsLegacy(true);

        UnitTester_Tensor tester = new UnitTester_Tensor("Testing core tensor functionality");

        Tsr<Double> x = Tsr.of(
                new int[]{2, 2},
                new double[]{
                        -1, 2,
                        -3, 3,
                }
        ).setRqsGradient(true);
        Tsr<Double> y = Tsr.of(new int[]{1, 1}, -3d);
        Tsr<Double> z = Tsr.of("I0xi1", x, y);
        tester.testTensor(z, new String[]{"[2x2]:(3.0, -6.0, 9.0, -9.0)"});
        z.backward(Tsr.of(new int[]{2, 2}, 1));
        tester.testTensor(x, new String[]{"[2x2]:(-1.0, 2.0, -3.0, 3.0):g:(-3.0, -3.0, -3.0, -3.0)"});
        //---
        {
            Neureka.get().settings().debug().setIsDeletingIntermediateTensors(false);
            x = Tsr.of(new int[]{1}, 0.1).setRqsGradient(true);
            Function tanh = new FunctionBuilder(Neureka.get().backend()).build("tanh(i0)", true);
            Function tenxx = new FunctionBuilder(Neureka.get().backend()).build("i0*100", true);
            z = tenxx.call(tanh.call(x));
            tester.testTensor(z, new String[]{"[1]:(9.95037E0)"});
            Neureka.get().settings().debug().setIsKeepingDerivativeTargetPayloads(true);
            z.backward(Tsr.of(new int[]{1}, 1d));
            tester.testTensor(x, new String[]{"[1]:(0.1):g:(99.0099E0)"});
            tester.testTensor(z, new String[]{"[1]:(9.95037E0); ->d[1]:(99.0099E0)"});
            tester.testContains(
                    z.toString("dgc"),
                    new String[]{"[1]:(9.95037E0); ->d[1]:(99.0099E0)"},
                    "test double formatting"
            );
            tester.testContains(
                    Tsr.of(3d).toString("dgc"),
                    new String[]{"[1]:(3.0)"},
                    "test FP formatting"
            );
        }
        {
            Neureka.get().settings().debug().setIsDeletingIntermediateTensors(true);
            x = Tsr.of(new int[]{1}, 0.1d).setRqsGradient(true);
            Function tanh = new FunctionBuilder(Neureka.get().backend()).build("tanh(i0)", true);
            Function tenxx = new FunctionBuilder(Neureka.get().backend()).build("i0*100", true);
            z = tenxx.call(tanh.call(x));
            tester.testTensor(z, new String[]{"[1]:(9.95037E0)"});
            Neureka.get().settings().debug().setIsKeepingDerivativeTargetPayloads(true);
            z.backward(Tsr.of(new int[]{1}, 1d));
            tester.testTensor(x, new String[]{"[1]:(0.1):g:(99.0099E0)"});
            tester.testTensor(z, new String[]{"[1]:(9.95037E0); ->ddeleted"});
            tester.testContains(
                    z.toString("dgc"),
                    new String[]{"[1]:(9.95037E0); ->ddelete"},
                    "test double formatting"
            );
            tester.testContains(
                    Tsr.of(3d).toString("dgc"),
                    new String[]{"[1]:(3.0)"},
                    "test FP formatting"
            );
        }
        Neureka.get().settings().debug().setIsKeepingDerivativeTargetPayloads(false);
        //---
        Tsr<Double> tensor1 = Tsr.of(3d).setRqsGradient(true);
        Tsr<Double> tensor2 = Tsr.of(-4d);
        Tsr<Double> tensor3 = Tsr.of(2d);
        tester.testInjection(
                new Tsr[]{tensor1, tensor2, tensor3},
                "I[2]<-(Ig[0]<-I[1])",
                new String[][]{
                        {"(-4.0)"},// == I[2]
                        {"(3.0)", "gdeleted"},//tensor1
                        {"(-4.0)"},//tensor2
                        {"(-4.0)"},//tensor3
                });
        tensor1 = Tsr.of(3d).setRqsGradient(true);
        tensor2 = Tsr.of(-4d);
        tensor3 = Tsr.of(2d);
        try {
            Function.of("I[2]<-(Ig[0]<-I[1])", true).call( tensor1, tensor2, tensor3 );
        } catch ( Exception e ) {
            tester.testContains(
                    e.getClass().getName()+" : "+e.getMessage(),
                    new String[]{"IllegalStateException", "'Function.create(\"left_inline(...)\", false)'", "Please use detached functions instead!"},
                    "Non-detached functions performing inline operations will throw exceptions on active autograd computation graphs!"
            );
        }
        Tsr<Double> result = Function.of("I[2]<-(Ig[0]<-I[1])", false).call( tensor1, tensor2, tensor3  );
        tester.testContains(
                result.toString(),
                new String[]{"(-4.0)"},
                "Testing if Function.setup.commit() returns non unique result!"
        );
        //---
        tensor1 = Tsr.of(new int[]{1, 3}, 2d);
        tensor2 = Tsr.of(new int[]{2, 1}, -1d);
        tensor1.setRqsGradient(true);
        tensor2.setRqsGradient(true);
        Neureka.get().settings().debug().setIsKeepingDerivativeTargetPayloads(true);
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1, tensor2},
                "relu(I[0]xI[1])",
                new String[]{
                        "[2x3]:(-0.02, -0.02, -0.02, -0.02, -0.02, -0.02);",
                        " =>d|[ [2x3]:(0.01, 0.01, 0.01, 0.01, 0.01, 0.01) ]|:t{ [2x3]:(-2.0, -2.0, -2.0, -2.0, -2.0, -2.0);",
                        " =>d|[ [2x1]:(-1.0, -1.0) ]|:t{ [1x3]:(2.0, 2.0, 2.0) }",
                        " =>d|[ [1x3]:(2.0, 2.0, 2.0) ]|:t{ [2x1]:(-1.0, -1.0) }"
                });
        //---
        tensor1 = Tsr.of(new int[]{1, 3}, 2d);
        tensor2 = Tsr.of(Double.class).withShape(2, 1).all(-1.0);
        tensor1.setRqsGradient(true);
        tensor2.setRqsGradient(true);
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1, tensor2},
                "lig((I[0]xI[1])*-100)",
                new String[]{
                        "[2x3]:(200.0, 200.0, 200.0, 200.0, 200.0, 200.0);",
                        " =>d|[ [2x3]:(-100.0, -100.0, -100.0, -100.0, -100.0, -100.0) ]|:t{ [2x3]:(-2.0, -2.0, -2.0, -2.0, -2.0, -2.0);",
                        " =>d|[ [1x3]:(2.0, 2.0, 2.0) ]|:t{ [2x1]:(-1.0, -1.0) }",
                        " =>d|[ [2x1]:(-1.0, -1.0) ]|:t{ [1x3]:(2.0, 2.0, 2.0) }",
                }
        );
        //---//Broken down to 2 functions:
        tensor1 = Tsr.of(new int[]{1, 3}, 2d);
        tensor2 = Tsr.of(new int[]{2, 1}, -1d);
        tensor1.setRqsGradient(true);
        tensor2.setRqsGradient(true);
        result = Tsr.of("I[0]xI[1]", tensor1, tensor2);
        result = Tsr.of("lig(I[0]*-100)", result);
        tester.testContains(result.toString("rc"),
                new String[]{
                        "[2x3]:(200.0, 200.0, 200.0, 200.0, 200.0, 200.0);",
                        " =>d|[ [2x3]:(-100.0, -100.0, -100.0, -100.0, -100.0, -100.0) ]|:t{ [2x3]:(-2.0, -2.0, -2.0, -2.0, -2.0, -2.0);",
                        " =>d|[ [1x3]:(2.0, 2.0, 2.0) ]|:t{ [2x1]:(-1.0, -1.0) }",
                        " =>d|[ [2x1]:(-1.0, -1.0) ]|:t{ [1x3]:(2.0, 2.0, 2.0) }"
                }, "");
        Neureka.get().settings().debug().setIsKeepingDerivativeTargetPayloads(false);
        //---
        tensor1 = Tsr.of(new int[]{2, 3, 4}, 2d);
        tensor1.setRqsGradient(false);
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1},//, tensor2},/<=TODO make this throw an exception (if input does not match function)
                "lig( [2, 1, 0]:( I[0] )*-100 )",
                new String[]{"[4x3x2]:(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)"}
        );
        //---
        tensor1 = Tsr.of(new int[]{2}, 2d);
        tensor2 = Tsr.of(new int[]{2}, 4d);
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
                                                ":t{ [2]:(4.0, 4.0) }",
                                        "=>d|[ [2]:(0.03112E0, 0.03112E0) ]|" +
                                                ":t{ [2]:(2.0, 2.0) }",
                                "}, ",
                        "=>d|[ [2]:(0.97996E0, 0.97996E0) ]|" +
                                ":t{ [2]:(4.0, 4.0) }"
                }
        );
        //---
        Neureka.get().settings().debug().setIsKeepingDerivativeTargetPayloads(true);
        tensor1 = Tsr.of(new int[]{3, 2, 1}, 4d);
        tensor2 = Tsr.of(Double.class).withShape(1, 1, 4).all(-1.0d);
        tensor3 = Tsr.of(new int[]{3, 2, 1}, 2d);
        tensor2.setRqsGradient(true);
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1, tensor2, tensor3},
                "I[0]xI[1]xI[2]",
                new String[]{
                        "[1x1x4]:(-48.0, -48.0, -48.0, -48.0);",
                        " =>d|[ [3x2x1]:(2.0, 2.0, 2.0, 2.0, 2.0, 2.0) ]|" +
                                ":t{ [3x2x4]:(-4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0); ",
                                    "=>d|[ [3x2x1]:(4.0, 4.0, 4.0, 4.0, 4.0, 4.0) ]|" +
                                        ":t{ [1x1x4]:(-1.0, -1.0, -1.0, -1.0) } " +
                                "}"
                });
        //--
        tensor1 = Tsr.of(new int[]{5, 1, 1}, 4d);//-2*4 = 8 | *3 = -24
        tensor2 = Tsr.of(new int[]{1, 4, 1}, -2d);
        tensor3 = Tsr.of(new int[]{1, 1, 2}, 3d);
        tensor1.setRqsGradient(true);
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1, tensor2, tensor3},
                "I[0]xI[1]xI[2]",
                new String[]{
                        "[5x4x2]:(-24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0); =>d|[ [1x1x2]:(3.0, 3.0) ]|:t{ [5x4x1]:(-8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0); =>d|[ [1x4x1]:(-2.0, -2.0, -2.0, -2.0) ]|:t{ [5x1x1]:(4.0, 4.0, 4.0, 4.0, 4.0) } }"
                }
        );
        Neureka.get().settings().debug().setIsKeepingDerivativeTargetPayloads(false);
        //=====================
        tensor1 = Tsr.ofDoubles().withShape(2,2).andFill(1.0, 2.0, 3.0, 4.0);
        tensor1.setRqsGradient(true);
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1},
                "((i0+2)^2)",
                new String[]{"d|[ [2x2]:(6.0, 8.0, 10.0, 12.0) ]|:t{ [2x2]:(1.0, 2.0, 3.0, 4.0) }"}
        );
        //---
        tensor1 = Tsr.of(new int[]{2, 2}, new double[]{1, 2, 3, 4});//-2*4 = 8 | *3 = -24
        tensor1.setRqsGradient(true);
        result = Tsr.of("(i0+2)", tensor1);
        result = Tsr.of("I[0]^2", result);
        tester.testContains(
                result.toString("rc"),
                new String[]{"d|[ [2x2]:(6.0, 8.0, 10.0, 12.0) ]|:t{ [2x2]:(1.0, 2.0, 3.0, 4.0) }"},
                ""
        );
        //=====================
        tensor1 = Tsr.of(new int[]{2}, new double[]{1, 2});//-2*4 = 8 | *3 = -24
        tensor1.setRqsGradient(true);
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1},
                "cos(tanh(lig(i0)))",
                new String[]{"[2]:(0.69985E0, 0.61771E0); =>d|[ [2]:(-0.19165E0, -0.12539E0) ]|:t{ [2]:(1.0, 2.0) }"}
        );
        //---
        //=====================
        tensor1 = Tsr.of(-3).setRqsGradient(true);
        tensor2 = Tsr.of(Double.class).scalar(4.0).setRqsGradient(true);
        tensor3 = Tsr.of(2);
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1, tensor2, tensor3},
                "(relu(i0*i1)+i1)/i2",
                new String[]{
                    "[1]:(1.94);",
                        "=>d|[ [1]:(0.5) ]|:",
                            "t{",
                                "[1]:(-0.12);",
                                    "=>d|[ [1]:(-0.03) ]|:",
                                        "t{ [1]:(4.0) }",
                                    "=>d|[ [1]:(0.04) ]|:",
                                        "t{ [1]:(-3.0) }",
                        "=>d|[ [1]:(0.5) ]|:",
                            "t{ [1]:(4.0) }"
                }
        );
        //=====================
        _subTest(tester);
        //=====================
        _subTest2(tester);
        //=====================


        // TESTING INVERSE:
        //=================

        tensor1 = Tsr.of(new int[]{2, 3}, new double[]{
                0, 0, 0,
                0, 0, 0,
        });
        //---
        tensor2 = Tsr.of(new int[]{5, 2}, new double[]{
                -2, 3,
                6, 3,
                -1, 0,
                2, 4,
                2, 1
        });
        tensor3 = Tsr.of(Double.class)
                        .withShape(4, 2)
                        .andFill(
                                1.0, 2.0,
                                -3.0, 2.0,//<= drain data!
                                4.0, -2.0,
                                -1.0, 5.0
                        );
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1, tensor2, tensor3},
                "i0<<xi1<<xi2",
                new String[]{
                        "[2x3]:(-26.0, 10.0, 32.0, 15.0, 34.0, 3.0)"
                }
        );

        assert tensor1.getNDConf() != null;
        assert tensor2.getNDConf() != null;
        assert tensor3.getNDConf() != null;

        result = Function.of("i0<<xi1<<xi2", true).call( tensor1, tensor2, tensor3 );
        tester.testContains(
                result.toString(),
                new String[]{"[2x3]:(-26.0, 10.0, 32.0, 15.0, 34.0, 3.0)"},
                "Testing if Function.setup.commit() returns non unique result!"
        );
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1, tensor2, tensor3},
                "i2x>>i1x>>i0",
                new String[]{"[2x3]:(-26.0, 10.0, 32.0, 15.0, 34.0, 3.0)"}
        );

        result =  Function.of("i2x>>i1x>>i0", true).call( tensor1, tensor2, tensor3 );
        tester.testContains(
                result.toString(),
                new String[]{"[2x3]:(-26.0, 10.0, 32.0, 15.0, 34.0, 3.0)"},
                "Testing if Function.setup.commit() returns non unique result!"
        );
        //=====================
        //---
        tensor1 = Tsr.ofDoubles()
                        .withShape(2, 2, 1)
                        .andFill(
                                1.0, 2.0, //3, 1,
                                2.0, -3.0 //-2, -1,
                        );
        tensor1.setRqsGradient(true);
        //---
        tensor2 = Tsr.of(new int[]{1, 2, 2}, new double[]{
                -2, 3,
                1, 2,
        });
        Neureka.get().settings().debug().setIsKeepingDerivativeTargetPayloads(true);
        tester.testTensorAutoGrad(//4, 5, -13, -4 <= result values
                new Tsr[]{tensor1, tensor2},
                "i0xi1",
                new String[]{"[2x1x2]:(0.0, 7.0, -7.0, 0.0); =>d|[ [1x2x2]:(-2.0, 3.0, 1.0, 2.0) ]|:t{ [2x2x1]:(1.0, 2.0, 2.0, -3.0) }"},
                Tsr.of(new int[]{2, 1, 2}, new double[]{1, 1, 1, 1}),
                new double[][]{{1.0, 3.0, 1.0, 3.0}, null}
        );
        Neureka.get().settings().debug().setIsKeepingDerivativeTargetPayloads(false);
        //---
        //======================
        int[] shape = {4, 2, 9, 5, 6, 2};
        int[] newForm = {1, 0, -1, -2, 2, 4, 3, -1, 5};
        int[] expected = {2, 4, 1, 2, 9, 6, 5, 1, 2};
        tester.testTensorUtility_reshape(shape, newForm, expected);
        //---
        shape = new int[]{4, 2, 9, 5, 6, 2};
        expected = new int[]{1080, 540, 60, 12, 2, 1, };
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
                        -4.0, 8.0, -6.0, -2.0,
                        4.0, -3.0, 8.0, -16.0,
                        12.0, 6.0, -12.0, 9.0
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
                        -10.0, 35.0, 7.0,
                        -16.0, 9.0, -52.0
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
                        -4.0, 0.0,
                        -13.0, -2.0
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
                        -26.0, 10.0, 32.0, 15.0, 34.0, 3.0
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
                        8.0, -4.0, 12.0, -4.0, 2.0, -6.0
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
                        14.0, -20.0, -2.0
                },
                true
        );
        //===========================================
        return true;
    }

    private static void _subTest(UnitTester_Tensor tester) {

        Neureka.get().settings().debug().setIsDeletingIntermediateTensors(false);

        Tsr<Double> tensor1 = Tsr.of(new int[]{1}, new double[]{2});//-2*4 = 8 | *3 = -24
        tensor1.setRqsGradient(true);
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1},//2 =>-2 =>-4 =>12 //2 =>-2 //-2,12 =>-24
                "(-3*(2*(i0*-1)))*(-1*i0)",
                new String[]{
                        "[1]:(-24.0); ",
                        "=>d|[ [1]:(12.0) ]|:" +
                                "t{ [1]:(-2.0); " +
                                "=>d|[ [1]:(-1.0) ]|:" +
                                "t{ [1]:(2.0) } " +
                                "}",
                        "=>d|[ [1]:(-2.0) ]|:" +
                                "t{ [1]:(12.0); " +
                                "=>d|[ [1]:(6.0) ]|:" +
                                "t{ [1]:(2.0) } " +
                                "}"
                }
        );

        Neureka.get().settings().debug().setIsDeletingIntermediateTensors(true);


        tensor1 = Tsr.of(new int[]{1}, new double[]{2});//-2*4 = 8 | *3 = -24
        tensor1.setRqsGradient(true);
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1},//2 =>-2 =>-4 =>12 //2 =>-2 //-2,12 =>-24
                "(-3*(2*(i0*-1)))*(-1*i0)",
                new String[]{
                        "[1]:(-24.0); ",
                        "=>d|[ [1]:(12.0) ]|:" +
                                "t{ deleted }",
                        "=>d|[ [1]:(-2.0) ]|:" +
                                "t{ [1]:(12.0); " +
                                "=>d|[ [1]:(6.0) ]|:" +
                                "t{ [1]:(2.0) } " +
                                "}"
                }
        );

    }


    private static void _subTest2(UnitTester_Tensor tester) {

        Neureka.get().settings().debug().setIsDeletingIntermediateTensors(false);
        Tsr<Double> tensor1 = Tsr.of(new int[]{1}, new double[]{2});//-2*4 = 8 | *3 = -24
        tensor1.setRqsGradient(true);
        Tsr<?> result = Tsr.of("(-3*(2*(i0*-1)))*(-1*i0)", tensor1);
        GraphNode<Double> node = (GraphNode) result.get( GraphNode.class );
        String asString = node.toString("gnv");
        tester.testContains(
                asString,
                new String[]{
                        "[1]:(-24.0)",
                        "[1]:(12.0)",
                        "[1]:(2.0)",
                        "[1]:(-3.0)",
                        "(-1.0 * I[0])",
                        "(I[0] * -1.0)",
                        "(I[0] * I[1])",
                        "LEAVE RQS GRADIENT",
                        "(I[0] * I[1]) => [1]:(-4.0)"
                },
                "Testing 'toString' of GraphNode");
        Neureka.get().settings().debug().setIsDeletingIntermediateTensors(true);
        result = Tsr.of("(-3*(2*(i0*-1)))*(-1*i0)", tensor1);
        node = (GraphNode) result.get( GraphNode.class );
        asString = node.toString("gnv");
        tester.testContains(
                asString,
                new String[]{
                        "[1]:(-24.0)",
                        "[1]:(12.0)",
                        "[1]:(2.0)",
                        "[1]:(-3.0)",
                        "(-1.0 * I[0])",
                        "(I[0] * -1.0)",
                        "(I[0] * I[1])",
                        "LEAVE RQS GRADIENT",
                        "deleted",
                        "(I[0] * -1.0) => deleted, type='BRANCH'"
                },
                "Testing 'toString' of GraphNode");
    }

}
