package neureka.unit;


import neureka.core.device.TDevice;
import neureka.core.T;
import neureka.unit.cases.NTester_Function;
import neureka.unit.cases.NTester_Tensor;
import neureka.unit.cases.NTester_TensorDevice;

public class NeurekaTest {

	public NeurekaTest(){}
	/*
	     TFunction IDs:
		 case 0:  ReLu
		 case 1:  Sigmoid
		 case 2:  Tanh
		 case 3:  Quadratic
		 case 4:  Ligmoid (rectifier)
		 case 5:  Linear
		 case 6:  Gaussian
	*/
	public void Test()
	{
	    testTensorFunctions();
		testTensorCore();
		testTensorDevice();
	}


	public void testTensorCore(){

		NTester_Tensor tester = new NTester_Tensor("Testing core tensor functionality");
		//---
		T tensor1 = new T(new int[]{1, 3}, 2);
		T tensor2 = new T(new int[]{2, 1}, -1);
		tensor1.setRqsGradient(true);
		tensor2.setRqsGradient(true);
		tester.testTensorAutoGrad(
				new T[]{tensor1, tensor2},
				"relu(I[0]xI[1])",
				new String[]{
						"[2x3]:(-0.02, -0.02, -0.02, -0.02, -0.02, -0.02);",
						" =>d|[ [2x3]:(0.01, 0.01, 0.01, 0.01, 0.01, 0.01) ]|:t{ [2x3]:(-2.0, -2.0, -2.0, -2.0, -2.0, -2.0);",
						" =>d|[ [2x1]:(-1.0, -1.0) ]|:t{ [1x3]:(2.0, 2.0, 2.0) },",
						" =>d|[ [1x3]:(2.0, 2.0, 2.0) ]|:t{ [2x1]:(-1.0, -1.0) },",
						"  }, "
				});
		//---
		tensor1 = new T(new int[]{1, 3}, 2);
		tensor2 = new T(new int[]{2, 1}, -1);
		tensor1.setRqsGradient(true);
		tensor2.setRqsGradient(true);
		tester.testTensorAutoGrad(
				new T[]{tensor1, tensor2},
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
		tensor1 = new T(new int[]{2, 3, 4}, 2);
		tensor1.setRqsGradient(false);
		tester.testTensorAutoGrad(
				new T[]{tensor1},//, tensor2},
				"lig([-2, 1, 0, -2]:(I[0])*-100)",
				new String[]{
						"[2x3x2x2]:(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)"
				}
		);
		//---
		tensor1 = new T(new int[]{2}, 2);
		tensor2 = new T(new int[]{2}, 4);
		tensor1.setRqsGradient(true);
		tensor2.setRqsGradient(true);
		tester.testTensorAutoGrad(
				new T[]{tensor1, tensor2},
				"lig(tanh(I[0]*I[1]*2)*I[1])",
				new String[]{
						"[2]:(4.010500886001868, 4.010500886001868); ",
								"=>d|[ [2]:(3.9275027410108176, 3.9275027410108176) ]|"+
									":t{ [2]:(0.9980525784828885, 0.9980525784828885); ",
									"=>d|[ [2]:(0.015564202334630739, 0.015564202334630739) ]|:t{ [2]:(4.0, 4.0) }, ",
									"=>d|[ [2]:(0.031128404669261478, 0.031128404669261478) ]|:t{ [2]:(2.0, 2.0) }, ",
									//	"=>d|[ [2]:(0.0077821011673153695, 0.0077821011673153695) ]|"+//TODO THIS IS INCORRECT!!!
									//		":t{ [2]:(4.0, 4.0) }, ",
									//	"=>d|[ [2]:(0.015564202334630739, 0.015564202334630739) ]|"+
									//		":t{ [2]:(2.0, 2.0) }, ",
									"}, ",
								"=>d|[ [2]:(0.9799635594161147, 0.9799635594161147) ]|"+
									":t{ [2]:(4.0, 4.0) }, "
				}
		);// [2]:(4.010500886001868, 4.010500886001868); =>d|[ [2]:(0.9799635594161147, 0.9799635594161147) ]|:t{ [2]:(4.0, 4.0) }, =>d|[ [2]:(3.9275027410108176, 3.9275027410108176) ]|:t{ [2]:(0.9980525784828885, 0.9980525784828885); =>d|[ [2]:(0.015564202334630739, 0.015564202334630739) ]|:t{ [2]:(4.0, 4.0) }, =>d|[ [2]:(0.031128404669261478, 0.031128404669261478) ]|:t{ [2]:(2.0, 2.0) },  },
		//---
		tensor1 = new T(new int[]{3, 2, 1}, 4);
		tensor2 = new T(new int[]{1, 1, 4}, -1);
		T tensor3 = new T(new int[]{3, 2, 1}, 2);
		tensor2.setRqsGradient(true);
		tester.testTensorAutoGrad(
				new T[]{tensor1, tensor2, tensor3},
				"I[0]xI[1]xI[2]",
				new String[]{
					"[1x1x4]:(-48.0, -48.0, -48.0, -48.0);",
						" =>d|[ [3x2x1]:(2.0, 2.0, 2.0, 2.0, 2.0, 2.0) ]|"+
							":t{ [3x2x4]:(-4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0); ",
						"=>d|[ [3x2x1]:(4.0, 4.0, 4.0, 4.0, 4.0, 4.0) ]|"+
							":t{ [1x1x4]:(-1.0, -1.0, -1.0, -1.0) },  }, "
				});
		//--
		tensor1 = new T(new int[]{5, 1, 1}, 4);//-2*4 = 8 | *3 = -24
		tensor2 = new T(new int[]{1, 4, 1}, -2);
		tensor3 = new T(new int[]{1, 1, 2}, 3);
		tensor1.setRqsGradient(true);
		tester.testTensorAutoGrad(
				new T[]{tensor1, tensor2, tensor3},
				"I[0]xI[1]xI[2]",
				new String[]{
						"[5x4x2]:(-24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0, -24.0); =>d|[ [1x1x2]:(3.0, 3.0) ]|:t{ [5x4x1]:(-8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0); =>d|[ [1x4x1]:(-2.0, -2.0, -2.0, -2.0) ]|:t{ [5x1x1]:(4.0, 4.0, 4.0, 4.0, 4.0) },  }, "
				}
		);
		//---
		int[] shape = {4, 2, 9, 5, 6, 2};
		int[] newForm = {1, 0, -1, -2, 2, 4, 3, -1, 5};
		int[] expected ={2, 4,  1,  2, 9, 6, 5, 1, 2};
		tester.testTensorUtility_reshape(shape, newForm, expected);
		//---
		shape = new int[]{4, 2, 9, 5, 6, 2};
		expected = new int[]{1, 4, 4*2, 4*2*9, 4*2*9*5, 4*2*9*5*6};
		tester.testTensorUtility_translation(shape, expected);
		//---
		shape = new int[]{4, 2, 9, 5, 6, 2};
		expected = new int[]{1, 1, 4, 0, 0, 0};
		tester.testTensorBase_idxFromAnchor(shape, 37, expected);
		//---
		shape = new int[]{4, 3, 2, 5};
		expected = new int[]{3, 2, 1, 2};
		int idx = (1)*3 + (1*4)*2 + (1*4*3)*1 + (1*4*3*2)*2;
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
		tester.testTensMulOn(
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
				4,-5,  2, 3,
		};
		scndShape = new int[]{2, 3};
		scndData = new double[]{
				4, -2,
				8, -1,
				-5, 2,
		};
		tester.testTensMulOn(
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
		tester.testTensMulOn(
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
		tester.testTensMulOn(
				frstShape, scndShape,
				frstData, scndData,
				new double[]{
						15, 11,
						20, -22,
				}
		);
	}

	public void testTensorDevice(){

		NTester_TensorDevice tester = new NTester_TensorDevice("Testing tensor device");
		TDevice gpu = new TDevice("nvidia");
		T tensor = T.factory.newTensor(new double[]{1, 3, 4, 2, -3, 2, -1, 6}, new int[]{2, 4});
		T firstTensor = tensor;
		tester.testAddTensor(gpu,tensor,
				new double[]{1,3,4,2,-3,2,-1,6},
				new int[]{2,4},
				new int[]{1,2},
				new int[]{0, 8, 0, 2, 0, 2});
		tensor = T.factory.newTensor(new double[]{-7,-9}, new int[]{2});
		tester.testAddTensor(gpu,tensor,
				new double[]{1,3,4,2,-3,2,-1,6, -7, -9, 0,0,0,0,0,0,0,0,0,0,},
				new int[]{2, 4},
				new int[]{1, 2},
				new int[]{0, 8, 0, 2, 0, 2,    8, 2, 0, 1, 0, 1});

		tensor = T.factory.newTensor(new double[]{4,-4,9,4, 77}, new int[]{5});
		tester.testAddTensor(gpu,tensor,
				new double[]{1,3,4,2,-3,2,-1,6, -7, -9, 4,-4,9,4,77,0,0,0,0,0,},
				new int[]{2, 4, 5, 0},
				new int[]{1, 2},
				new int[]{0, 8, 0, 2, 0, 2,    8, 2, 0, 1, 0, 1,   10, 5, 2, 1, 0, 1});
		tester.testGetTensor(gpu, firstTensor,
				new double[]{1,3,4,2,-3,2,-1,6, -7, -9, 4,-4,9,4,77,0,0,0,0,0,},
				new int[]{2, 4, 5, 0},
				new int[]{1, 2},
				new int[]{8, 2, 0, 1, 0, 1,   10, 5, 2, 1, 0, 1}
				);
		tester.testAddTensor(gpu, firstTensor,
				new double[]{1,3,4,2,-3,2,-1,6, -7, -9, 4,-4,9,4,77,0,0,0,0,0,},
				new int[]{2, 4, 5, 0},
				new int[]{1, 2},
				new int[]{0, 8, 0, 2, 0, 2,    8, 2, 0, 1, 0, 1,   10, 5, 2, 1, 0, 1}
		);
		tester.testGetTensor(gpu, firstTensor,
				new double[]{1,3,4,2,-3,2,-1,6, -7, -9, 4,-4,9,4,77,0,0,0,0,0,},
				new int[]{2, 4, 5, 0},
				new int[]{1, 2},
				new int[]{8, 2, 0, 1, 0, 1,   10, 5, 2, 1, 0, 1}
		);
		tensor = T.factory.newTensor(new double[]{888, 777, -33, 999}, new int[]{2, 2});
		tensor.setRqsGradient(true);
		tester.testAddTensor(gpu, tensor,
				new double[]{888,777,-33,999,0,0,0,0, -7, -9, 4,-4,9,4,77,0,0,0,0,0,},
				new int[]{2, 4, 5, 2, 2, 0, 0, 0},
				new int[]{1, 2},
				new int[]{0, -4, 3, 2, 0, 2,    8, 2, 0, 1, 0, 1,   10, 5, 2, 1, 0, 1}
		);

		//TESTING TENS MUL ON GPU NOW!!!!!
		T src1 = T.factory.newTensor(
				new double[]{
						1, 2,
						3, 4,
						0, 2,

						3, 4,
						2, -1,
						-2, -3
				},
				new int[]{2, 3, 2}
				);
		T src2 = T.factory.newTensor(
				new double[]{
						-1, 3,
						0, 2,

						-3, 1,
						2, -3,

						0, 4,
						5, -1
				}, new int[]{2, 2, 3}
				);
		T drn = T.factory.newTensor(//0, 5, 3, 6, -3, 3, 5, 1, 2, 3, 3 -4
				new double[]{
						0, 0,
						0, 0,
				}, new int[]{1, 2, 2}
				);
		System.out.println("pre mul:");
		System.out.println(gpu.stringified(gpu.getKernel().values()));
		gpu.add(src1);//<= make these tests! ?
		gpu.add(src2);//...
		gpu.add(drn);//...
		//System.out.println("THIS IS HAPPENING:");
		//gpu.printDeviceContent(true);
		//gpu.calculate_on_CPU(drn, src1, src2, 18);
		//gpu.printDeviceContent(true);
		tester.testCalculation(
				gpu,
				drn,src1, src2,  18,//Tsr mul
				new double[]{888.0, 777.0, -33.0, 999.0, 0.0, 0.0, 0.0, 0.0, -7.0, -9.0, 4.0, -4.0, 9.0, 4.0, 77.0, 1.0, 2.0, 3.0, 4.0, 0.0, 2.0, 3.0, 4.0, 2.0, -1.0, -2.0, -3.0, -1.0, 3.0, 0.0, 2.0, -3.0, 1.0, 2.0, -3.0, 0.0, 4.0, 5.0, -1.0, 15.0, 11.0, 20.0, -22.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
		);
		tester.testCalculation(
				gpu,
				drn,src1, src2,  18,//Tsr mul
				new double[]{888.0, 777.0, -33.0, 999.0, 0.0, 0.0, 0.0, 0.0, -7.0, -9.0, 4.0, -4.0, 9.0, 4.0, 77.0, 1.0, 2.0, 3.0, 4.0, 0.0, 2.0, 3.0, 4.0, 2.0, -1.0, -2.0, -3.0, -1.0, 3.0, 0.0, 2.0, -3.0, 1.0, 2.0, -3.0, 0.0, 4.0, 5.0, -1.0, 15.0, 11.0, 20.0, -22.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
		);
		System.out.println("WE ARE HERE:");
		gpu.printDeviceContent(true);

		addition:
		drn = T.factory.newTensor(
				new double[]{
						111, 222, 333,
						444, 555, 666,
						777, 888, 999,
						1098, 32, 150,
				}, new int[]{3, 2, 2}
		);
		//Adding new drain:
		tester.testAddTensor(gpu, drn,
				new double[]{888.0, 777.0, -33.0, 999.0, 0.0, 0.0, 0.0, 0.0, -7.0, -9.0, 4.0, -4.0, 9.0, 4.0, 77.0, 1.0, 2.0, 3.0, 4.0, 0.0, 2.0, 3.0, 4.0, 2.0, -1.0, -2.0, -3.0, -1.0, 3.0, 0.0, 2.0, -3.0, 1.0, 2.0, -3.0, 0.0, 4.0, 5.0, -1.0, 15.0, 11.0, 20.0, -22.0, 111.0, 222.0, 333.0, 444.0, 555.0, 666.0, 777.0, 888.0, 999.0, 1098.0, 32.0, 150.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, },
				new int[]{2, 4, 5, 2, 2, 3, 2, 1, 2, 2, 3, 2, 2, 0, 0, 0, 0,},
				new int[]{1, 2, 1, 2, 6, 1, 2, 4, 1, 1, 2, 1, 3, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,},
				new int[]{0, -4, 3, 2, 0, 2, 8, 2, 0, 1, 0, 1, 10, 5, 2, 1, 0, 1, 15, 12, 4, 3, 2, 3, 27, 12, 3, 3, 5, 3, 39, 4, 7, 3, 8, 3, 43, 12, 10, 3, 11, 3, }
		);
		//gpu.printDeviceContent(true);
		tester.testCalculation(
				gpu,
				drn, src1, src2, 17,//Tsr add
				new double[]{888.0, 777.0, -33.0, 999.0, 0.0, 0.0, 0.0, 0.0, -7.0, -9.0, 4.0, -4.0, 9.0, 4.0, 77.0, 1.0, 2.0, 3.0, 4.0, 0.0, 2.0, 3.0, 4.0, 2.0, -1.0, -2.0, -3.0, -1.0, 3.0, 0.0, 2.0, -3.0, 1.0, 2.0, -3.0, 0.0, 4.0, 5.0, -1.0, 15.0, 11.0, 20.0, -22.0, 0.0, 5.0, 3.0, 6.0, -3.0, 3.0, 5.0, 1.0, 2.0, 3.0, 3.0, -4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,}
		);
		//gpu.calculate_on_CPU(drn, src1, src2, 17);
		//gpu.calculate_on_CPU(drn, drn, null, 6);
		tester.testCalculation(
				gpu,
				drn, drn, null, 6,//Tsr gaus
				new double[]{888.0, 777.0, -33.0, 999.0, 0.0, 0.0, 0.0, 0.0, -7.0, -9.0, 4.0, -4.0, 9.0, 4.0, 77.0, 1.0, 2.0, 3.0, 4.0, 0.0, 2.0, 3.0, 4.0, 2.0, -1.0, -2.0, -3.0, -1.0, 3.0, 0.0, 2.0, -3.0, 1.0, 2.0, -3.0, 0.0, 4.0, 5.0, -1.0, 15.0, 11.0, 20.0, -22.0, 1.0, 1.3887943864964039E-11, 1.2340980408667962E-4, 2.3195228302435736E-16, 1.2340980408667962E-4, 1.2340980408667962E-4, 1.3887943864964039E-11, 0.36787944117144233, 0.018315638888734182, 1.2340980408667962E-4, 1.2340980408667962E-4, 1.1253517471925921E-7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, }
		);
		//System.out.println("is shared: "+gpu.device.isSharedMemory());
		//for(int i=0; i<2000; i++){
		//	try {
		//		Thread.sleep(20);
		//	} catch (InterruptedException e) {
		//		e.printStackTrace();
		//	}
		//	gpu.add(T.factory.newTensor(1+i, new int[]{100000}));//...
		//}
		//try {
		//	Thread.sleep(3000);
		//} catch (InterruptedException e) {
		//	e.printStackTrace();
		//}
		gpu.getKernel().dispose();
		System.out.println("Done!");
	}

	public int testTensorFunctions()
	{
		NTester_Function tester = new NTester_Function("Testing function factory and scalar calculations");
		int TestCounter = 0;
		int SuccessCounter = 0;

		//EXPRESSION TESTING:
		TestCounter++;
		SuccessCounter += tester.testExpression("sum(ij)", "sum(I[j])", "");
		tester.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");

		TestCounter++;
		SuccessCounter += tester.testExpression("sum(1*(4-2/ij))", "sum(1.0*(4.0-(2.0/I[j])))", "");
		tester.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");

		TestCounter++;
		SuccessCounter += tester.testExpression("quadratic(ligmoid(Ij))", "quad(lig(I[j]))", "");
		tester.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");

		TestCounter++;
		SuccessCounter += tester.testExpression("softplus(I[3]^(3/i1)/sum(Ij^2)-23+I0/i1)", "lig((((I[3]^(3.0/I[1]))/sum(I[j]^2.0))-23.0)+(I[0]/I[1]))", "");
		tester.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");

		TestCounter++;
		SuccessCounter += tester.testExpression("1+3+5-23+I0*45/(345-651^I3-6)", "(1.0+3.0+(5.0-23.0)+(I[0]*(45.0/(345.0-(651.0^I[3])-6.0))))", "");
		tester.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");

		TestCounter++;
		SuccessCounter += tester.testExpression("sin(23*i1)-cos(i0^0.3)+tanh(23)", "((sin(23.0*I[1])-cos(I[0]^0.3))+tanh(23.0))", "");
		tester.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");

		TestCounter++;
		SuccessCounter += tester.testExpression("2*3/2-1", "((2.0*(3.0/2.0))-1.0)", "");
		tester.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");

		TestCounter++;
		SuccessCounter += tester.testExpression("3x5xI[4]xI[3]", "(((3.0x5.0)xI[4])xI[3])", "");
		tester.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");

		TestCounter++;
		SuccessCounter += tester.testExpression("[1,0, 5,3, 4]:(tanh(i0xi1))", "([1,0,5,3,4]:(tanh(I[0]xI[1])))", "");
		tester.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");

		TestCounter++;
		SuccessCounter += tester.testExpression("[0,2, 1,3, -1](sig(I0))", "([0,2,1,3,-1]:(sig(I[0])))", "");
		tester.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");

		//ACTIVATION TESTING:
		TestCounter++;
		double[] input1 = {};
		SuccessCounter += tester.testActivation("6/2*(1+2)", input1, 9, "");
		tester.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");

		TestCounter++;
		input1 = new double[]{2,3.2,6};
		SuccessCounter += tester.testActivation("sum(Ij)", input1, 11.2, "");
		tester.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");

		TestCounter++;
		double[] input2 = {0.5,0.5,100};
		SuccessCounter += tester.testActivation("prod(Ij)", input2, 25, "");
		tester.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");

		TestCounter++;
		double[] input3 = {0.5,0.5,10};
		SuccessCounter += tester.testActivation("prod(prod(Ij))", input3, (2.5*2.5*2.5), "");
		tester.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");

		TestCounter++;
		double[] input4 = {5,4,3,12};//12/4-5+2+3
		SuccessCounter += tester.testActivation("I3/i[1]-I0+2+i2", input4, (3), "");
		tester.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");

		TestCounter++;
		double[] input5 = {-4,-2,6,-3,-8};//-3*-2/(-8--4-2)
		SuccessCounter += tester.testActivation("i3*i1/(i4-i0-2)-sig(0)+tanh(0)", input5, (-1.5), "");
		tester.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");


		TestCounter++;
		T[] tsrs = new T[]{
				T.factory.newTensor(new double[]{1,2}, new int[]{2}),
				T.factory.newTensor(new double[]{3,-4}, new int[]{2})
		};
		T expected = T.factory.newTensor(new double[]{0.9701425001453319, -0.8944271909999159}, new int[]{2});
		SuccessCounter += tester.testActivation("tanh(sum(Ij))", tsrs, expected, "");
		tester.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");

		TestCounter++;
		expected = T.factory.newTensor(new double[]{0.3132616875182228, 0.6931471805599453}, new int[]{2});
		SuccessCounter += tester.testActivation("lig(prod(Ij-2))", tsrs, expected, "");
		tester.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");

		TestCounter++;
		tsrs = new T[]{
				T.factory.newTensor(new double[]{10, 12, 16, 21, 33, 66, 222, 15}, new int[]{2, 4})
		};
		expected = T.factory.newTensor(new double[]{8.000335406372896, 10.000045398899218, 14.000000831528373, 19.000000005602796, 31.000000000000036, 64.0, 220.0, 13.000002260326852}, new int[]{1, 2, 2, 2});
		SuccessCounter += tester.testActivation("lig([-1, 0, -2, -2](Ij-2))", tsrs, expected, "");
		tester.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");



		tester.println("###########################");
		tester.println("|| FUNCTION END RESULT:");
		tester.println("|| ============>> "+SuccessCounter+"/"+TestCounter);
		tester.println("###########################");

		return 0;
	}

	
	
}
