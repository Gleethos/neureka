package st.tests;

import neureka.Tsr;
import neureka.devices.Device;
import neureka.devices.opencl.OpenCLDevice;
import testutility.UnitTester_Tensor;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public class CrossDeviceSystemTest
{

    public static boolean on( Device gpu )
    {
        int initialNumberOfOutsourced = gpu.size();

        UnitTester_Tensor tester = new UnitTester_Tensor("");

        List<Tsr> listOfTensors = new ArrayList<>();
        Tsr<Double> tensor1, tensor2;
        //=====================================================================
        tensor1 = Tsr.of(new int[]{2, 2}, new double[]{
                -1, 7,
                -2, 3,
        }).setRqsGradient(true);
        tensor2 = Tsr.of(new int[]{2, 2}, new double[]{
                -1, 7,
                -2, 3,
        });
        gpu.store(tensor1).store(tensor2);
        listOfTensors.add(tensor1);
        listOfTensors.add(tensor2);

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
        tensor1 = Tsr.of(new int[]{3, 5}, new double[]{
                2,  3,  5,
                -4,  6,  2,
                -5, -2, -1,
                2,  4, -1,
                1,  2,  7
        });
        gpu.store(tensor1);
        listOfTensors.add(tensor1);
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1}, "lig(I[0])",
                new String[]{
                        "[3x5]:(2.12693E0, 3.04859E0, 5.00672E0, 0.01814E0, 6.00248E0, 2.12693E0, 0.00671E0, 0.12692E0, 0.31326E0, 2.12693E0, 4.01815E0, 0.31326E0, 1.31326E0, 2.12693E0, 7.00091E0)"
                });
        //===================
        tensor1 = Tsr.of(new int[]{2}, 3);
        tensor2 = Tsr.of(new int[]{2}, 4);
        gpu.store(tensor1).store(tensor2);
        listOfTensors.add(tensor1);
        listOfTensors.add(tensor2);
        //result = Tsr.of(new Tsr[]{tensor1, tensor2}, "i0*i1");
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1, tensor2}, "i0*i1",
                new String[]{"[2]:(12.0, 12.0)"});
        //===========================================
        tensor1 = Tsr.of(
                new int[]{2, 3, 1},
                new double[]{
                        3, 2,
                        -1, -2,
                        2, 4
                }
        );
        tensor2 = Tsr.of(
                new int[]{1, 3, 2},
                new double[]{
                        4, -1, 3,
                        2, 3, -1
                });
        gpu.store(tensor1);
        gpu.store(tensor2);
        listOfTensors.add(tensor1);
        listOfTensors.add(tensor2);
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1, tensor2}, "I0 x i1",
                new String[]{
                        "[2x1x2]:(15.0, 2.0, 10.0, 2.0)"
                });
        //=======================
        tensor1 = Tsr.of(
                new int[]{200, 300, 1},
                2
        );
        tensor2 = Tsr.of(
                new int[]{1, 300, 200},
                3);
        gpu.store(tensor1);
        gpu.store(tensor2);
        listOfTensors.add(tensor1);
        listOfTensors.add(tensor2);
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1, tensor2}, "I0xi1",
                new String[]{
                        "[200x1x200]:(1800.0, 1800.0, 1800.0, 1800.0, 1800.0, 1800.0,"//...
                });
        //---
        tensor1 = Tsr.of(new int[]{2, 2, 1}, new double[]{
                1,  2, //  3, 1,
                2, -3, // -2, -1,
        }).setRqsGradient(true);
        tensor2 = Tsr.of(new int[]{1, 2, 2}, new double[]{
                -2, 3, //  0  7
                1, 2,  // -7  0
        });
        gpu.store(tensor1).store(tensor2);
        listOfTensors.add(tensor1);
        listOfTensors.add(tensor2);
        tester.testTensorAutoGrad(//4, 5, -13, -4 <= result values
                new Tsr[]{tensor1, tensor2},
                "i0xi1",
                new String[]{
                        "[2x1x2]:(" +
                                "0.0, 7.0, -7.0, 0.0"+
                                "); =>d|[ [1x2x2]:(-2.0, 3.0, 1.0, 2.0) ]|:t{ [2x2x1]:(1.0, 2.0, 2.0, -3.0) }"
                },
                Tsr.of(new int[]{2, 1, 2}, new double[]{1, 1, 1, 1}),
                new double[][]{new double[]{1.0, 3.0, 1.0, 3.0}, null}
        );

        // ---
        Tsr<Double> x = Tsr.of(new int[]{1}, 3d).setRqsGradient(true);
        Tsr<Double> b = Tsr.of(new int[]{1}, -4d);
        Tsr<Double> w = Tsr.of(new int[]{1}, 2d);
        gpu.store(x).store(b).store(w);
        listOfTensors.add(x);
        listOfTensors.add(b);
        listOfTensors.add(w);
        /*
         *      ((3-4)*2)^2 = 4
         *  dx:   8*3 - 32  = -8
         */
        Tsr<Double> y = Tsr.of("((i0+i1)*i2)^2", x, b, w);
        tester.testTensor(y, new String[]{"[1]:(4.0); ->d[1]:(-8.0)"});
        y.backward(Tsr.of(2));
        tester.testTensor(x, new String[]{"-16.0"});
        tester.testShareDevice(gpu, new Tsr[]{y, x, b, w});
        //---
        x = Tsr.of(new int[]{1}, 4).setRqsGradient(true);
        b = Tsr.of(new int[]{1}, 0.5);
        w = Tsr.of(new int[]{1}, 0.5);
        gpu.store(x).store(b).store(w);
        listOfTensors.add(x);
        listOfTensors.add(b);
        listOfTensors.add(w);
        y = Tsr.of("(2^i0^i1^i2^2", x, b, w);
        tester.testTensor(y, new String[]{"[1]:(4.0);", " ->d[1]:(1.38629E0)"});
        tester.testShareDevice(gpu, new Tsr[]{y, x, b, w});

        //====
        x = Tsr.of(new int[]{1}, 3);
        b = Tsr.of(new int[]{1}, -5);
        w = Tsr.of(new int[]{1}, -2);
        gpu.store(x).store(b).store(w);
        listOfTensors.add(x);
        listOfTensors.add(b);
        listOfTensors.add(w);
        Tsr z = Tsr.of("I0*i1*i2", x, b, w);
        tester.testTensor(z, new String[]{"[1]:(30.0)"});
        tester.testShareDevice(gpu, new Tsr[]{z, x, b, w});

        //---
        x = Tsr.of(new int[]{1}, 3).setRqsGradient(true);
        b = Tsr.of(new int[]{1}, 0.5);
        w = Tsr.of(new int[]{1}, 4);
        gpu.store(x).store(b).store(w);
        listOfTensors.add(x);
        listOfTensors.add(b);
        listOfTensors.add(w);
        y = Tsr.of("(12/i0/i1/i2/2", x, b, w);//12/3/0.5/4/2 .... 12 * 1/ (0.0625)
        //listOfTensors.add(y);
        tester.testTensor(y, new String[]{"[1]:(1.0);", " ->d[1]:(-0.33333E0)"});
        tester.testShareDevice(gpu, new Tsr[]{y, x, b, w});
        //---
        //---------------------------------------------
        y = null;
        z = null;
        x = null;
        b = null;
        w = null;
        tensor1 = null;
        tensor2 = null;
        listOfTensors.forEach((t)->t.setRqsGradient(false));//Removes gradients!
        System.gc();
        try { Thread.sleep(400); } catch (InterruptedException e) { e.printStackTrace(); }
        System.gc();
        try { Thread.sleep(400); } catch (InterruptedException e) { e.printStackTrace(); }

        if(gpu instanceof OpenCLDevice)
        {
            Collection<Tsr> outsourced = gpu.getTensors();
            int numberOfOutsourced = outsourced.size() - initialNumberOfOutsourced;

            String sentence = "Number of outsourced tensors: ";
            tester.testContains(
                    sentence + numberOfOutsourced,
                    new String[]{sentence+(listOfTensors.size())},
                    "Testing for memory leaks!"
            );
            //---
            boolean[] stillOnDevice = new boolean[]{true};
            listOfTensors.forEach((t)->stillOnDevice[0] = outsourced.contains(t)&&stillOnDevice[0]);
            sentence = "Used tensors still on device: ";
            tester.testContains(
                    sentence +stillOnDevice[0],
                    new String[]{sentence+"true"},
                    "Testing for memory leaks!"
            );
            //---
            listOfTensors.forEach(gpu::free);
            sentence = "Number of tensors after deleting: ";
            tester.testContains(
                    sentence + Math.max(0, gpu.getTensors().size()-initialNumberOfOutsourced),
                    new String[]{sentence+"0"},
                    "Testing if all tensors have been deleted!"
            );
            //---
        }
        return true;
    }


}
