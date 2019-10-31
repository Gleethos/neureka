
import neureka.core.Tsr;
import neureka.core.device.IDevice;
import neureka.core.device.aparapi.AparapiDevice;
import neureka.core.device.aparapi.KernelFP32;
import neureka.core.device.aparapi.KernelFP64;
import neureka.core.device.openCL.OpenCLDevice;
import neureka.core.device.openCL.OpenCLPlatform;
import org.junit.Test;
import util.NTester_Tensor;
import util.NTester_TensorDevice;

public class TestOnGPU {

    @Test
    public void testTesting(){
        //AparapiDevice gpu = new AparapiDevice("intel gpu FP64");
        //Tsr x = new Tsr(new int[]{300, 300, 1}, 2);
        //Tsr y = new Tsr(new int[]{1, 300, 300}, -1);
        //gpu.add(x).add(y);
        //System.out.println("hey!");
        //Tsr z = null;
        //for(int i=0; i<100; i++){
        //    z = new Tsr(x, "x", y);
        //    gpu.rmv(z);
        //}
    //    System.out.println(z);
    //   //JOCLSimpleMandelbrot.start(null);
    //   org.jocl.samples.MyJOCLMandelbrot.start(null);
         //new Setup().initCL();
       /////OpenCLDevice myDevice = OpenCLPlatform.PLATFORMS().get(0).getDevices().get(0);
       /////System.out.println("Local mem size: "+myDevice.localMemSize());
       ///// System.out.println("name: "+myDevice.name());
       ///// System.out.println("vendor: "+myDevice.vendor());

    //   try {
    //       Thread.sleep(7000000);
    //   } catch (InterruptedException e) {
    //       e.printStackTrace();
    //   }
    }

    @Test
    public void testAutograd(){
        if(!System.getProperty("os.name").toLowerCase().contains("windows")){
            return;
        }
        NTester_Tensor tester = new NTester_Tensor("Testing autograd on GPU");

        //---
        IDevice gpu = new AparapiDevice("intel gpu");//FP32 ought to be chosen by default here!
        tester.testContains(
                (((AparapiDevice)gpu).getKernel() instanceof KernelFP32)?"FP32":"FP64",
                new String[]{"FP32"},
                "Test device kernel FP-Type");
        _testAutograd(gpu, tester);
        //---
        gpu = new AparapiDevice("intel gpu FP64");
        tester.testContains(
                (((AparapiDevice)gpu).getKernel() instanceof KernelFP64)?"FP62":"FP34",
                new String[]{"FP34"},
                "Test device kernel FP-Type");
        _testAutograd(gpu, tester);
        //---
        gpu = OpenCLPlatform.PLATFORMS().get(0).getDevices().get(0);
        //_testAutograd(gpu, tester);

        tester.close();
    }
    private  void _testAutograd(IDevice gpu, NTester_Tensor tester){
        Tsr tensor1, tensor2;
        //=====================================================================
        tensor1 = new Tsr(new int[]{2, 2}, new double[]{
                -1, 7,
                -2, 3,
        }).setRqsGradient(true);//ERROR!! ???
        tensor2 = new Tsr(new int[]{2, 2}, new double[]{
                -1, 7,
                -2, 3,
        });
        gpu.add(tensor1).add(tensor2);
        //System.out.println(new Tsr(t, "lig(I[0])"));
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1, tensor2},
                "I[0]*i1",
                new String[]{
                        "[2x2]:(1.0, 49.0, 4.0, 9.0)"
                });
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1, tensor2},
                "I[0]xi1",
                new String[]{
                        "[1x1]:(63.0)",
                        "[2x2]:(-1.0, 7.0, -2.0, 3.0) ]|:t{ [2x2]:(-1.0, 7.0, -2.0, 3.0) }"
                });
        //=====================================================================
        tensor1 = new Tsr(new int[]{3, 5}, new double[]{
                 2,  3,  5,
                -4,  6,  2,
                -5, -2, -1,
                 2,  4, -1,
                 1,  2,  7
        });
        gpu.add(tensor1);
        //System.out.println(new Tsr(t, "lig(I[0])"));
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1},
                "lig(I[0])",
                new String[]{
                        "[3x5]:(2.12693E0, 3.04859E0, 5.00672E0, 0.01814E0, 6.00248E0, 2.12693E0, 0.00671E0, 0.12692E0, 0.31326E0, 2.12693E0, 4.01815E0, 0.31326E0, 1.31326E0, 2.12693E0, 7.00091E0)"
                });
        //===================
        tensor1 = new Tsr(new int[]{2}, 3);
        tensor2 = new Tsr(new int[]{2}, 4);
        gpu.add(tensor1).add(tensor2);
        //result = new Tsr(new Tsr[]{tensor1, tensor2}, "i0*i1");
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
        gpu.add(x).add(b).add(w);
        y = new Tsr(new Tsr[]{x, b, w}, "(2^i0^i1^i2^2");
        tester.testTensor(y, new String[]{"[1]:(4.0);", " ->d[1]:(1.38629E0), "});
        tester.testShareDevice(gpu, new Tsr[]{y, x, b, w});

        //====
        x = new Tsr(new int[]{1}, 3);
        b = new Tsr(new int[]{1}, -5);
        w = new Tsr(new int[]{1}, -2);
        gpu.add(x).add(b).add(w);
        Tsr z = new Tsr(new Tsr[]{x, b, w}, "I0*i1*i2");
        tester.testTensor(z, new String[]{"[1]:(30.0)"});
        tester.testShareDevice(gpu, new Tsr[]{z, x, b, w});
    }

    @Test
    public void testTensorDevice() {
        if(!System.getProperty("os.name").toLowerCase().contains("windows")){
            return;
        }
        NTester_TensorDevice tester = new NTester_TensorDevice("Testing tensor device");

        AparapiDevice gpu = new AparapiDevice("intel gpu FP32");
        tester.testContains(
                (gpu.getKernel() instanceof KernelFP32)?"FP32":"FP64",
                new String[]{"FP32"},
                "Test device kernel FP-Type");
        _testTensorDevice(gpu, tester);

        gpu = new AparapiDevice("intel gpu FP64");
        tester.testContains(
                (gpu.getKernel() instanceof KernelFP64)?"FP62":"FP34",
                new String[]{"FP34"},
                "Test device kernel FP-Type");
        _testTensorDevice(gpu, tester);

        tester.close();
    }

    private void _testTensorDevice(AparapiDevice gpu, NTester_TensorDevice tester){
        Tsr tensor = Tsr.fcn.newTsr(new double[]{1, 3, 4, 2, -3, 2, -1, 6}, new int[]{2, 4});
        Tsr firstTensor = tensor;
        tester.testAddTensor(gpu, tensor,
                new double[]{1, 3, 4, 2, -3, 2, -1, 6},
                new int[]{2, 4},
                new int[]{1, 2},
                new int[]{0, 8, 0, 2, 0, 2});
        tensor = Tsr.fcn.newTsr(new double[]{-7, -9}, new int[]{2});
        tester.testAddTensor(gpu, tensor,
                new double[]{1, 3, 4, 2, -3, 2, -1, 6, -7, -9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,},
                new int[]{2, 4},
                new int[]{1, 2},
                new int[]{0, 8, 0, 2, 0, 2, 8, 2, 0, 1, 0, 1});
        tensor = Tsr.fcn.newTsr(new double[]{4, -4, 9, 4, 77}, new int[]{5});
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
        tensor = Tsr.fcn.newTsr(new double[]{888, 777, -33, 999}, new int[]{2, 2});
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
                drn, src1, src2, 17, -1,//Tsr overwrite64
                new double[]{888.0, 777.0, -33.0, 999.0, 0.0, 0.0, 0.0, 0.0, -7.0, -9.0, 4.0, -4.0, 9.0, 4.0, 77.0, 1.0, 2.0, 3.0, 4.0, 0.0, 2.0, 3.0, 4.0, 2.0, -1.0, -2.0, -3.0, -1.0, 3.0, 0.0, 2.0, -3.0, 1.0, 2.0, -3.0, 0.0, 4.0, 5.0, -1.0, 15.0, 11.0, 20.0, -22.0, -8.0, 4.0, -9.0, -2.0, 2.0, 3.0, -2.0, 3.0, 6.0, 3.0, -1.0, 0.0, 2.0, 4.0, 2.0, 1.0, 1.0, 2.0, -3.0, 2.0, 4.0, -2.0, -1.0, 5.0, 0.0, 5.0, 3.0, 6.0, -3.0, 3.0, 5.0, 1.0, 2.0, 3.0, 3.0, -4.0,}
        );
        tester.testCalculation(
                gpu,
                drn, drn, null, 6, -1,//Tsr gaus
                new double[]{888.0, 777.0, -33.0, 999.0, 0.0, 0.0, 0.0, 0.0, -7.0, -9.0, 4.0, -4.0, 9.0, 4.0, 77.0, 1.0, 2.0, 3.0, 4.0, 0.0, 2.0, 3.0, 4.0, 2.0, -1.0, -2.0, -3.0, -1.0, 3.0, 0.0, 2.0, -3.0, 1.0, 2.0, -3.0, 0.0, 4.0, 5.0, -1.0, 15.0, 11.0, 20.0, -22.0, -8.0, 4.0, -9.0, -2.0, 2.0, 3.0, -2.0, 3.0, 6.0, 3.0, -1.0, 0.0, 2.0, 4.0, 2.0, 1.0, 1.0, 2.0, -3.0, 2.0, 4.0, -2.0, -1.0, 5.0, 1.0, 1.3887943864964039E-11, 1.2340980408667962E-4, 2.3195228302435736E-16, 1.2340980408667962E-4, 1.2340980408667962E-4, 1.3887943864964039E-11, 0.36787944117144233, 0.018315638888734182, 1.2340980408667962E-4, 1.2340980408667962E-4, 1.1253517471925921E-7,}
        );
        //---
        gpu.getKernel().dispose();
    }


}