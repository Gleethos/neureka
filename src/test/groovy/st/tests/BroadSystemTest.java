package st.tests;

import neureka.Neureka;
import neureka.Shape;
import neureka.Tensor;
import neureka.autograd.GraphNode;
import neureka.math.Function;
import testutility.UnitTester_Tensor;

public class BroadSystemTest
{

    public static boolean on()
    {
        Neureka.get().settings().view().getNDPrintSettings().setIsLegacy(true);

        UnitTester_Tensor tester = new UnitTester_Tensor("Testing core tensor functionality");

        Tensor<Double> x = Tensor.of(Shape.of(2, 2), new double[]{
                        -1d, 2d,
                        -3d, 3d
                    }).setRqsGradient(true);
        Tensor<Double> y = Tensor.of(Shape.of(1, 1), -3d);
        Tensor<Double> z = Tensor.of("I0xi1", x, y);
        tester.testTensor(z, new String[]{"[2x2]:(3.0, -6.0, 9.0, -9.0)"});
        z.backward(Tensor.of(Shape.of(2, 2), 1d));
        tester.testTensor(x, new String[]{"[2x2]:(-1.0, 2.0, -3.0, 3.0):g:(-3.0, -3.0, -3.0, -3.0)"});
        //---
        {
            Neureka.get().settings().debug().setIsDeletingIntermediateTensors(false);
            x = Tensor.of(Shape.of(1), 0.1d).setRqsGradient(true);
            Function tanh = Function.of("tanh(i0)", true);
            Function tenxx = Function.of("i0*100", true);
            z = tenxx.call(tanh.call(x));
            tester.testTensor(z, new String[]{"[1]:(9.9668)"});
            Neureka.get().settings().debug().setIsKeepingDerivativeTargetPayloads(true);
            z.backward(Tensor.of(Shape.of(1), 1d));
            tester.testTensor(x, new String[]{"[1]:(0.1):g:(99.0066)"});
            tester.testTensor(z, new String[]{"[1]:(9.9668); ->d[1]:(99.0066)"});
            tester.testContains(
                    z.toString("dgc"),
                    new String[]{"[1]:(9.9668); ->d[1]:(99.0066)"},
                    "test double formatting"
            );
            tester.testContains(
                    Tensor.of(3d).toString("dgc"),
                    new String[]{"[1]:(3.0)"},
                    "test FP formatting"
            );
        }
        {
            Neureka.get().settings().debug().setIsDeletingIntermediateTensors(true);
            x = Tensor.of(Shape.of(1), 0.1d).setRqsGradient(true);
            Function tanh = Function.of("tanh(i0)", true);
            Function tenxx = Function.of("i0*100", true);
            z = tenxx.call(tanh.call(x));
            tester.testTensor(z, new String[]{"[1]:(9.9668)"});
            Neureka.get().settings().debug().setIsKeepingDerivativeTargetPayloads(true);
            z.backward(Tensor.of(Shape.of(1), 1d));
            tester.testTensor(x, new String[]{"[1]:(0.1):g:(99.0066)"});
            tester.testTensor(z, new String[]{"[1]:(9.9668); ->d[1]:(99.0066)"});
            tester.testContains(
                    z.toString("dgc"),
                    new String[]{"[1]:(9.9668); ->d[1]:(99.0066)"},
                    "test double formatting"
            );
            tester.testContains(
                    Tensor.of(3d).toString("dgc"),
                    new String[]{"[1]:(3.0)"},
                    "test FP formatting"
            );
        }
        Neureka.get().settings().debug().setIsKeepingDerivativeTargetPayloads(false);
        //---
        Tensor<Double> tensor1 = Tensor.of(3d).setRqsGradient(true);
        Tensor<Double> tensor2 = Tensor.of(-4d);
        Tensor<Double> tensor3 = Tensor.of(2d);
        tester.testInjection(
                new Tensor[]{tensor1, tensor2, tensor3},
                "I[2]<-(Ig[0]<-I[1])",
                new String[][]{
                        {"(-4.0)"},// == I[2]
                        {"(3.0)", "g:(-4.0)"},//tensor1
                        {"(-4.0)"},//tensor2
                        {"(-4.0)"},//tensor3
                });
        tensor1 = Tensor.of(3d).setRqsGradient(true);
        tensor2 = Tensor.of(-4d);
        tensor3 = Tensor.of(2d);
        try {
            Function.of("I[2]<-(Ig[0]<-I[1])", true).call( tensor1, tensor2, tensor3 );
        } catch ( Exception e ) {
            tester.testContains(
                    e.getClass().getName()+" : "+e.getMessage(),
                    new String[]{"IllegalStateException", "'Function.create(\"left_inline(...)\", false)'", "Please use detached functions instead!"},
                    "Non-detached functions performing inline operations will throw exceptions on active autograd computation graphs!"
            );
        }
        Tensor<Double> result = Function.of("I[2]<-(Ig[0]<-I[1])", false).call( tensor1, tensor2, tensor3  );
        tester.testContains(
                result.toString(),
                new String[]{"(-4.0)"},
                "Testing if Function.setup.commit() returns non unique result!"
        );
        //---
        tensor1 = Tensor.of(Shape.of(1, 3), 2d);
        tensor2 = Tensor.of(Shape.of(2, 1), -1d);
        tensor1.setRqsGradient(true);
        tensor2.setRqsGradient(true);
        Neureka.get().settings().debug().setIsKeepingDerivativeTargetPayloads(true);
        tester.testTensorAutoGrad(
                new Tensor[]{tensor1, tensor2},
                "relu(I[0]xI[1])",
                new String[]{
                        "[2x3]:(-0.02, -0.02, -0.02, -0.02, -0.02, -0.02);",
                        " =>d|[ [2x3]:(0.01, 0.01, 0.01, 0.01, 0.01, 0.01) ]|:t{ [2x3]:(-2.0, -2.0, -2.0, -2.0, -2.0, -2.0);",
                        " =>d|[ [2x1]:(-1.0, -1.0) ]|:t{ [1x3]:(2.0, 2.0, 2.0) }",
                        " =>d|[ [1x3]:(2.0, 2.0, 2.0) ]|:t{ [2x1]:(-1.0, -1.0) }"
                });
        //---
        //---//Broken down to 2 functions:
        tensor1 = Tensor.of(Shape.of(1, 3), 2d);
        tensor2 = Tensor.of(Shape.of(2, 1), -1d);
        tensor1.setRqsGradient(true);
        tensor2.setRqsGradient(true);
        result = Tensor.of("I[0]xI[1]", tensor1, tensor2);
        result = Tensor.of("lig(I[0]*-100)", result);
        tester.testContains(result.toString("rc"),
                new String[]{
                        "[2x3]:(200.0, 200.0, 200.0, 200.0, 200.0, 200.0);",
                        " =>d|[ [2x3]:(-100.0, -100.0, -100.0, -100.0, -100.0, -100.0) ]|:t{ [2x3]:(-2.0, -2.0, -2.0, -2.0, -2.0, -2.0);",
                        " =>d|[ [1x3]:(2.0, 2.0, 2.0) ]|:t{ [2x1]:(-1.0, -1.0) }",
                        " =>d|[ [2x1]:(-1.0, -1.0) ]|:t{ [1x3]:(2.0, 2.0, 2.0) }"
                }, "");
        Neureka.get().settings().debug().setIsKeepingDerivativeTargetPayloads(false);
        //---
        tensor1 = Tensor.of(Shape.of(2, 3, 4), 2d);
        tensor1.setRqsGradient(false);
        tester.testTensorAutoGrad(
                new Tensor[]{tensor1},//, tensor2},/<=TODO make this throw an exception (if input does not match function)
                "lig( [2, 1, 0]:( I[0] )*-100 )",
                new String[]{"[4x3x2]:(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)"}
        );
        //---
        //---
        Neureka.get().settings().debug().setIsKeepingDerivativeTargetPayloads(true);
        tensor1 = Tensor.of(Shape.of(3, 2, 1), 4d);
        tensor2 = Tensor.of(Double.class).withShape(1, 1, 4).all(-1.0d);
        tensor3 = Tensor.of(Shape.of(3, 2, 1), 2d);
        tensor2.setRqsGradient(true);
        tester.testTensorAutoGrad(
                new Tensor[]{tensor1, tensor2, tensor3},
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
        tensor1 = Tensor.of(Shape.of(5, 1, 1), 4d);//-2*4 = 8 | *3 = -24
        tensor2 = Tensor.of(Shape.of(1, 4, 1), -2d);
        tensor3 = Tensor.of(Shape.of(1, 1, 2), 3d);
        tensor1.setRqsGradient(true);
        tester.testTensorAutoGrad(
                new Tensor[]{tensor1, tensor2, tensor3},
                "I[0]xI[1]xI[2]",
                new String[]{
                        "[5x4x2]:(-24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0); =>d|[ [1x1x2]:(3.0, 3.0) ]|:t{ [5x4x1]:(-8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0); =>d|[ [1x4x1]:(-2.0, -2.0, -2.0, -2.0) ]|:t{ [5x1x1]:(4.0, 4.0, 4.0, 4.0, 4.0) } }"
                }
        );
        Neureka.get().settings().debug().setIsKeepingDerivativeTargetPayloads(false);
        //=====================
        tensor1 = Tensor.ofDoubles().withShape(2,2).andFill(1.0, 2.0, 3.0, 4.0);
        tensor1.setRqsGradient(true);
        tester.testTensorAutoGrad(
                new Tensor[]{tensor1},
                "((i0+2)**2)",
                new String[]{"d|[ [2x2]:(6.0, 8.0, 10.0, 12.0) ]|:t{ [2x2]:(1.0, 2.0, 3.0, 4.0) }"}
        );
        //---
        tensor1 = Tensor.of(Shape.of(2, 2), new double[]{1d, 2d, 3d, 4d});//-2*4 = 8 | *3 = -24
        tensor1.setRqsGradient(true);
        result = Tensor.of("(i0+2)", tensor1);
        result = Tensor.of("I[0]**2", result);
        tester.testContains(
                result.toString("rc"),
                new String[]{"d|[ [2x2]:(6.0, 8.0, 10.0, 12.0) ]|:t{ [2x2]:(1.0, 2.0, 3.0, 4.0) }"},
                ""
        );
        //=====================
        tensor1 = Tensor.of(Shape.of(2), new double[]{1d, 2d});//-2*4 = 8 | *3 = -24
        tensor1.setRqsGradient(true);
        tester.testTensorAutoGrad(
                new Tensor[]{tensor1},
                "cos(tanh(lig(i0)))",
                new String[]{"[2]:(0.64856, 0.56366); =>d|[ [2]:(-0.14000, -0.04020) ]|:t{ [2]:(1.0, 2.0) }"}
        );
        //---
        //=====================
        tensor1 = Tensor.of(-3d).setRqsGradient(true);
        tensor2 = Tensor.of(Double.class).scalar(4.0).setRqsGradient(true);
        tensor3 = Tensor.of(2d);
        tester.testTensorAutoGrad(
                new Tensor[]{tensor1, tensor2, tensor3},
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

        // TESTING INVERSE:
        //=================

        tensor1 = Tensor.of(Shape.of(2, 3), new double[]{
                0d, 0d, 0d,
                0d, 0d, 0d
            });
        //---
        tensor2 = Tensor.of(Shape.of(5, 2), new double[]{
                -2d, 3d,
                 6d, 3d,
                -1d, 0d,
                 2d, 4d,
                 2d, 1d
            });
        tensor3 = Tensor.of(Double.class)
                        .withShape(4, 2)
                        .andFill(
                                1.0, 2.0,
                                -3.0, 2.0,//<= drain data!
                                4.0, -2.0,
                                -1.0, 5.0
                        );
        tester.testTensorAutoGrad(
                new Tensor[]{tensor1, tensor2, tensor3},
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
                new Tensor[]{tensor1, tensor2, tensor3},
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
        tensor1 = Tensor.ofDoubles()
                        .withShape(2, 2, 1)
                        .andFill(
                                1.0, 2.0, //3, 1,
                                2.0, -3.0 //-2, -1,
                        );
        tensor1.setRqsGradient(true);
        //---
        tensor2 = Tensor.of(Shape.of(1, 2, 2), new double[]{
                            -2d, 3d,
                            1d, 2d
                    });
        Neureka.get().settings().debug().setIsKeepingDerivativeTargetPayloads(true);
        tester.testTensorAutoGrad(//4, 5, -13, -4 <= result values
                new Tensor[]{tensor1, tensor2},
                "i0xi1",
                new String[]{"[2x1x2]:(0.0, 7.0, -7.0, 0.0); =>d|[ [1x2x2]:(-2.0, 3.0, 1.0, 2.0) ]|:t{ [2x2x1]:(1.0, 2.0, 2.0, -3.0) }"},
                Tensor.of(Shape.of(2, 1, 2), new double[]{1, 1, 1, 1}),
                new double[][]{{1.0, 3.0, 1.0, 3.0}, null}
        );
        Neureka.get().settings().debug().setIsKeepingDerivativeTargetPayloads(false);
        //---
        //======================
        int[] shape = {4, 2, 9, 5, 6, 2};
        int[] newForm = {1, 0, -1, -2, 2, 4, 3, -1, 5};
        int[] expected = {2, 4, 1, 2, 9, 6, 5, 1, 2};
        tester.testTensorUtility_permute(shape, newForm, expected);
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
                Shape.of(frstShape), Shape.of(scndShape),
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
                Shape.of(frstShape), Shape.of(scndShape),
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
                Shape.of(frstShape), Shape.of(scndShape),
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
                Shape.of(frstShape), Shape.of(scndShape),
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
                Shape.of(frstShape), Shape.of(scndShape),
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
                Shape.of(frstShape), Shape.of(scndShape),
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
        tester.testInvTensBroadcast(
                Shape.of(frstShape), Shape.of(scndShape),
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
}
