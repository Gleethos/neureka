package neureka.unit;


import neureka.main.core.NVertex;
import neureka.main.core.base.data.RelativeGradients;
import neureka.main.core.base.data.T;
import neureka.main.core.imp.NVertex_Root;
import neureka.main.core.modul.calc.TDevice;
import neureka.unit.io.NVTesting_Activation;
import neureka.unit.state.NVTesting_Format;
import neureka.unit.state.NVTesting_Tensor;
import neureka.unit.state.NVTesting_TensorDevice;
import neureka.utility.NMessageFrame;

public class NVTester {

	NMessageFrame Console = new NMessageFrame("NV-Unit-Test:");
	NMessageFrame ResultConsole = new NMessageFrame("NV-Unit-Test-Result:");
	
	public NVTester(){}
	/*
	     Function IDs:
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
		//NVertex source = new NVertex_Root(6, true, false, -1, 3);
		//NVertex head = new NVertex_Root(4, false, false, 1, 2);
		//source.randomizeInput();
		//head.connect(source);
		//head.weightAll();
		//head.randomizeBiasAndWeight();
		//System.out.println(head.toString());

		//InputFormattingTest();
	    //BaseFunctionsActivationTest();
	    //testCoreTypes();
		//advancedActivationTest();
		//networkTest();
	    componentTest();
		//rootActivationTest();
		//simpleRootTest();
		//testRootDeriviation();
		//weightStateTest();
		testTensorCore();

		testTensorDevice();
	}


	public void testTensorCore(){

		NVTesting_Tensor tester = new NVTesting_Tensor(Console, ResultConsole);
		//---
		T tensor1 = new T(new int[]{1, 3}, 2);
		T tensor2 = new T(new int[]{2, 1}, -1);
		tensor1.setRqsGradient(true);
		tensor2.setRqsGradient(true);
		RelativeGradients derivatives = new RelativeGradients();
		derivatives.put(tensor1, T.factory.newTensor(new double[]{-0.01,-0.01,-0.01,-0.01,-0.01,-0.01}, new int[]{2, 3}));
		derivatives.put(tensor2, T.factory.newTensor(new double[]{0.02,0.02,0.02,0.02,0.02,0.02}, new int[]{2, 3}));
		T expectedTensor = T.factory.newTensor(new double[]{-0.02,-0.02,-0.02,-0.02,-0.02,-0.02}, new int[]{2, 3});
		expectedTensor.addModule(derivatives);
		tester.testTensorAutoGrad(
				new T[]{tensor1, tensor2},
				"relu(tensmul(Ij))",
				expectedTensor
		);
		//---
		tensor1 = new T(new int[]{1, 3}, 2);
		tensor2 = new T(new int[]{2, 1}, -1);
		tensor1.setRqsGradient(true);
		tensor2.setRqsGradient(true);
		derivatives = new RelativeGradients();//200.0, 200.0, 200.0, 200.0, 200.0, 200.0); ->d[2x3]:(-200.0, -200.0, -200.0, -200.0, -200.0, -200.0), ->d[2x3]:(100.0, 100.0, 100.0, 100.0, 100.0, 100.0),
		derivatives.put(tensor1, T.factory.newTensor(new double[]{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}, new int[]{2, 3}));
		derivatives.put(tensor2, T.factory.newTensor(new double[]{-200.0, -200.0, -200.0, -200.0, -200.0, -200.0}, new int[]{2, 3}));
		expectedTensor = T.factory.newTensor(new double[]{200, 200, 200, 200, 200, 200}, new int[]{2, 3});
		expectedTensor.addModule(derivatives);
		tester.testTensorAutoGrad(
				new T[]{tensor1, tensor2},
				"lig((I[0]xI[1])*-100)",
				expectedTensor
		);
		//---
		tensor1 = new T(new int[]{3, 2, 1}, 4);
		tensor2 = new T(new int[]{1, 1, 4}, -1);
		T tensor3 = new T(new int[]{3, 2, 1}, 2);
		tensor2.setRqsGradient(true);
		derivatives = new RelativeGradients();
		derivatives.put(tensor1, T.factory.newTensor(new double[]{48,48,48,48}, new int[]{1, 1, 4}));
		expectedTensor = T.factory.newTensor(new double[]{-48,-48,-48,-48}, new int[]{1, 1, 4});
		expectedTensor.addModule(derivatives);
		tester.testTensorAutoGrad(new T[]{tensor1, tensor2, tensor3}, "tensmul", expectedTensor);
		//--
		tensor1 = new T(new int[]{5, 1, 1}, 4);//-2*4 = 8 | *3 = -24
		tensor2 = new T(new int[]{1, 4, 1}, -2);
		tensor3 = new T(new int[]{1, 1, 2}, 3);
		tensor1.setRqsGradient(true);
		derivatives = new RelativeGradients();
		derivatives.put(tensor1, new T(new int[]{5, 4, 2}, -6));
		expectedTensor = new T(new int[]{5, 4, 2}, -24);
		expectedTensor.addModule(derivatives);
		tester.testTensorAutoGrad(new T[]{tensor1, tensor2, tensor3}, "tensmul", expectedTensor);
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

		NVTesting_TensorDevice tester = new NVTesting_TensorDevice(Console, ResultConsole);
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
		gpu.add(src1);
		gpu.add(src2);
		gpu.add(drn);
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


	}

	public void testRootDeriviation() {
		NVTesting_Activation tester = new NVTesting_Activation(Console, ResultConsole);
		int TestCounter = 0;
		int SuccessCounter = 0;
		NVertex parent = new NVertex_Root(1, false, false, 0, 4);//ligmoid
		NVertex child = new NVertex_Root(parent, true, false, -1, 3);//quadratic!
        NVertex src = new NVertex_Root(1,true, false, -10, 0);//relu

        child.connect(src);
		parent.connect(child);

		src.setAllInput(0, -220);//2.2 * 5 -> 10.5; 10.5*10.5 -> 105.25; -> 105.25...
		parent.setWeight(0, 0, 0, 5);//-220 =>(relu)=> -2.2 =(W=null)=> -2.2 =>

		src.asExecutable().loadLatest();
		src.asExecutable().forward();
		src.asExecutable().loadLatest();

		NVertex[] structure = {parent, child};
		double[] expected = {24.20000000003091};
		double[][] relational = {{-21.99999999932002}};
		SuccessCounter += tester.testGraph_RelationalDeriviation(structure, parent, expected, relational);
		TestCounter++;
		Console.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");
	}
	
	public void simpleRootTest() 
	{
		//Hidden:
		NVertex SuperRoot = new NVertex_Root(1, false, false, 0, 0);
		NVertex.State.turnIntoParentRoot(SuperRoot);
		NVertex basicSub = new NVertex_Root(1, true, false, 1, 0);
		NVertex.State.turnIntoBasicChild(basicSub, SuperRoot);
		
		NVertex src = new NVertex_Root(1, true, false, -1, 0);
		src.setAllInput(0, -3);
		src.asExecutable().loadLatest();
		src.asExecutable().forward();
		
		basicSub.connect(src);
		basicSub.setWeight(0, 0, 0, -2);
		
		SuperRoot.connect(basicSub);
		SuperRoot.setWeight(0, 0, 0, 2);
		
		basicSub.addToAllInput(0, 4);
		System.out.println("\nStart:");
		
		SuperRoot.asExecutable().loadLatest();
		basicSub.asExecutable().loadLatest();
		
		SuperRoot.asExecutable().forward();
		basicSub.asExecutable().forward();
		
		Console.println("SuperRoot:");
		Console.println(SuperRoot.toString());
		Console.println("SubRoot:");
		Console.println(basicSub.toString());
	}
	
	public int[] neuralActivationTest() 
	{	
		NVTesting_Activation tester = new NVTesting_Activation(Console, ResultConsole);
		
		int TestCounter = 0;
		int SuccessCounter = 0;
		System.out.println("\nTEST : "+TestCounter+"\n======================================================================================");
		NVertex head = new NVertex_Root(1, true, false, 0, 0);
		NVertex child = new NVertex_Root(1, false, false, 1, 0);
		
		head.connect(child);
		child.setAllInput(0, 2);
		NVertex[] net = {head, child};
		TestCounter++;
		SuccessCounter+=tester.testGraph_ScalarActivation(net, head, 2, "Activating and being first layer:");
		Console.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");
		//-----------------------------------------------------------
		
		return null;
	}
	public int[] ActivationTest() 
	{	
		NVTesting_Activation tester = new NVTesting_Activation(Console, ResultConsole);
		
		int TestCounter = 0;
		int SuccessCounter = 0;
		System.out.println("\nTEST : "+TestCounter+"\n======================================================================================");
		NVertex head = new NVertex_Root(1, true, false, 0, 0);
		NVertex child = new NVertex_Root(1, false, false, 1, 0);
		
		
		head.connect(child);
		child.setAllInput(0, 2);
		NVertex[] net = {head, child};
		TestCounter++;
		SuccessCounter+=tester.testGraph_ScalarActivation(net, head, 2, "(Root!) Activating and being first layer:");
		Console.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");
		//-----------------------------------------------------------
		
		return null;
	}
	
	public int[] rootActivationTest() 
	{
		NVTesting_Activation tester = new NVTesting_Activation(Console, ResultConsole);
		
		int TestCounter = 0;
		int SuccessCounter = 0;
		System.out.println("\nTEST : "+TestCounter+"\n======================================================================================");
		NVertex mother = new NVertex_Root(1, true, false, 0, 0);
		NVertex child = new NVertex_Root(1, false, false, 1, 0);
		
		NVertex.State.turnIntoChildOfParent(child, mother);
		mother.connect(child);
		child.setAllInput(0, 2);
		NVertex[] net = {mother, child};
		TestCounter++;
		SuccessCounter+=tester.testGraph_ScalarActivation(net, mother, 2, "Activating and being first layer:");
		Console.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");
		//-----------------------------------------------------------
		System.out.println("\nTEST : "+TestCounter+"\n======================================================================================");
		mother = new NVertex_Root(1,false, false, 0, 0);
		child = new NVertex_Root(mother.asNode(),false, false, 1, 0);
		
		//NVertex.State.turnIntoChildOfMother(child, mother);
		mother.connect(child);
		child.setAllInput(0, 2);
		net = new NVertex[2]; net[0]=mother; net[1]=child;
		TestCounter++;
		SuccessCounter+=tester.testGraph_ScalarActivation(net, mother, 0, "Activating without being first layer, last layer or being connected to first layer:");
		
		Console.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");
		//-----------------------------------------------------------
		System.out.println("\nTEST : "+TestCounter+"\n======================================================================================");
		mother = new NVertex_Root(1,false, false, 0, 0);
		child = new NVertex_Root(mother.asNode(),false, false, 1, 0);
		
		//NVertex.State.turnIntoChildOfMother(child, mother);
		mother.connect(child);
		child.setAllInput(0, -21);
		net = new NVertex[2]; net[0]=mother; net[1]=child;
		TestCounter++;
		SuccessCounter+=tester.testGraph_ScalarActivation(net, mother, 0, "Activating without being first layer, last layer or being connected to first layer:");
		
		Console.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");
		//-----------------------------------------------------------
		System.out.println("\nTEST : "+TestCounter+"\n======================================================================================");
		mother = new NVertex_Root(1,false, false, 0, 0);
		child = new NVertex_Root(1 ,false, false, 1, 0);
				
		NVertex.State.turnIntoChildOfParent(child, mother);
		mother.connect(child);
		child.setAllInput(0, 3);
		net = new NVertex[2]; net[0]=mother; net[1]=child;
		TestCounter++;
		SuccessCounter+=tester.testGraph_ScalarActivation(net, mother, 0, "Activating without being first layer or being connected to first layer: (using NVertex.State.turnIntoChildOfMother(child, mother);)");
		
		Console.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");
		//-----------------------------------------------------------
		System.out.println("\nTEST : "+TestCounter+"\n======================================================================================");
		mother = new NVertex_Root(1,true, false, 0, 0);
		child = new NVertex_Root(1,false, false, 1, 0);
		
		mother.connect(child);
		NVertex.State.turnIntoChildOfParent(child, mother);

		child.setAllInput(0, 2);
		net = new NVertex[2]; net[0]=mother; net[1]=child;
		TestCounter++;
		SuccessCounter+=tester.testGraph_ScalarActivation(net, mother, 0,"Activating on root whose components were connected as both mothers:");
		Console.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");
		//-----------------------------------------------------------
		System.out.println("\nTEST : "+TestCounter+"\n======================================================================================");
		Console.println("Checking if weight is e_set of network whose components were connected as both mothers:");
		mother = new NVertex_Root(1,true, false, 0, 0);
		child = new NVertex_Root(1,false, false, 1, 0);
		
		mother.connect(child);
		NVertex.State.turnIntoChildOfParent(child, mother);
		
		double[] w = mother.getWeight(0, 0);
		if(w!=null) {SuccessCounter++;}
		TestCounter++;
		Console.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");
		//-----------------------------------------------------------
		System.out.println("\nTEST : "+TestCounter+"\n======================================================================================");

		mother = new NVertex_Root(1,true, false, 0, 0);
		child = new NVertex_Root(1,false, false, 1, 0);
				
		NVertex.State.turnIntoChildOfParent(child, mother);
		mother.connect(child);
		child.setAllInput(0, -2);
		net = new NVertex[2]; net[0]=mother; net[1]=child;
		TestCounter++;
		SuccessCounter+=tester.testGraph_ScalarActivation(net, mother, -0.0002, "Activating on root with two cascading relu nodes and negative input (-2->-0.02->-0.0002):");
		Console.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");
		//-----------------------------------------------------------
		System.out.println("\nTEST : "+TestCounter+"\n======================================================================================");
		mother = new NVertex_Root(1,true, false, 0, 0);
		child = new NVertex_Root(1,false, false, 1, 0);
				
		mother.connect(child);
		NVertex.State.turnIntoChildOfParent(child, mother);
		mother.setWeight(0, 0, 0, -1.0);
		child.setAllInput(0, -2);
		net = new NVertex[2]; net[0]=mother; net[1]=child;
		TestCounter++;
		SuccessCounter+=tester.testGraph_ScalarActivation(net, mother, +0.02, "Activating on root with two cascading relu nodes and negative input and negative weight(-2->-0.02->+0.02):");
		Console.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");
		
		//-----------------------------------------------------------
		System.out.println("\nTEST : "+TestCounter+"\n======================================================================================");
		NVertex source = new NVertex_Root(1,true, false, -1, 0);
		
		mother = new NVertex_Root(1,true, false, 0, 0);
		child = new NVertex_Root(1,false, false, 1, 0);
		
		source.setAllInput(0, 4);
		source.asExecutable().loadLatest();
		source.asExecutable().forward();
		source.asExecutable().loadLatest();
		
		NVertex.State.turnIntoChildOfParent(child, mother);	
		mother.connect(child);
		child.connect(source);
		child.setWeight(0, 0, 0, 1.0);
		//child.setInput(0, -2);
		net = new NVertex[2]; net[0]=mother; net[1]=child;
		TestCounter++;
		SuccessCounter+=tester.testGraph_ScalarActivation(net, mother, +4, "Activating on root connected to first layer neuron (setting weight to 1):");
		Console.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");
		//-----------------------------------------------------------
		System.out.println("\nTEST : "+TestCounter+"\n======================================================================================");
		Console.println("Checking if an input root will retain weight upon connecting to neuron:");
		source = new NVertex_Root(1,true, false, -1, 0);
		
		mother = new NVertex_Root(1,true, false, 0, 0);
		child = new NVertex_Root(1,false, false, 1, 0);
		
		source.setAllInput(0, 4);
		source.asExecutable().loadLatest();
		source.asExecutable().forward();
		source.asExecutable().loadLatest();
		
		NVertex.State.turnIntoChildOfParent(child, mother);	
		mother.connect(child);
		child.connect(source);
		//mother.setWeight(-1, 0, 0);
		child.setWeight(0, 0, 0, 2.5);
		w = child.getWeight(0, 0); 
		
		//net = new NVertex[2]; net[0]=mother; net[1]=child;
		TestCounter++;
		//SuccessCounter+=this.testRootActivation(net, mother, +8);
		if(w!=null) {if(w[0]==2.5) {SuccessCounter++;}}
		Console.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");
		//-----------------------------------------------------------
		System.out.println("\nTEST : "+TestCounter+"\n======================================================================================");
		source = new NVertex_Root(1,true, false, -1, 0);
		mother = new NVertex_Root(1,true, false, 1, 0);
		child = new NVertex_Root(1,false, false, 0, 0);
		
		source.setAllInput(0, 3);
		source.asExecutable().loadLatest();
		source.asExecutable().forward();
		source.asExecutable().loadLatest();
		
		NVertex.State.turnIntoChildOfParent(child, mother);	
		mother.connect(child);
		child.connect(source);
		//mother.setWeight(-1, 0, 0);
		child.setWeight(0, 0, 0, 2.0);
		net = new NVertex[2]; net[0]=mother; net[1]=child;
		//net = new NVertex[2]; net[0]=mother; net[1]=child;
		
		SuccessCounter+=tester.testGraph_ScalarActivation(net, mother, 6, "Checking if an input root will retain weight upon connecting to neuron:");
		Console.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");
	    System.out.println(source.getActivation(0)+" ...\n"+child.toString());
		//-----------------------------------------------------------
		System.out.println("\nTEST : "+TestCounter+"\n======================================================================================");
		Console.println("Checking root derivatrives:");
		
		source = new NVertex_Root(1,true, false, -1, 0);
		
		mother = new NVertex_Root(1,true, false, 0, 0);
		child = new NVertex_Root(1,false, false, 1, 0);
		NVertex otherChild = new NVertex_Root(1,false, false, 2, 0);

		NVertex.State.turnIntoChildOfParent(child, mother);	
		NVertex.State.turnIntoChildOfParent(otherChild, mother);
		mother.connect(child);
		mother.connect(otherChild);
		child.setAllInput(0, 2.5);
		otherChild.setAllInput(0, 2);

		net = new NVertex[3]; net[0]=mother; net[1]=child; net[2]=otherChild;
		TestCounter++; double[] derivatives = {1,1,1};
		SuccessCounter+=tester.testGraph_InputDeriviation(net, mother, derivatives);
		Console.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");
		//-----------------------------------------------------------
		System.out.println("\nTEST : "+TestCounter+"\n======================================================================================");
		Console.println("Checking root derivatrives:");
		
		source = new NVertex_Root(1,true, false, -1, 0);
		
		mother     = new NVertex_Root(1,true, false, 0, 0);
		child      = new NVertex_Root(mother.asNode(),false, false, 1, 0);
		otherChild = new NVertex_Root(mother.asNode(),false, false, 2, 0);

		NVertex.State.turnIntoChildOfParent(child, mother);	
		NVertex.State.turnIntoChildOfParent(otherChild, mother);
		mother.connect(child);
		mother.connect(otherChild);
		child.setAllInput(0, 2.5);
		otherChild.setAllInput(0, 2);

		double[] weight = {2, 3};
		mother.setWeight(0,0,weight);
		
		net = new NVertex[3]; 
		net[0]=mother; net[1]=child; net[2]=otherChild;
		TestCounter++; 
		derivatives[0]=1;
		derivatives[0]=2;
		derivatives[0]=3;
		SuccessCounter+=tester.testGraph_InputDeriviation(net, mother, derivatives);
		Console.println("###########################");
		Console.println("|| ROOT STRUCTURE ACTIVATION RESULT:");
		Console.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");
		Console.println("###########################");
		//-----------------------------------------------------------	
		ResultConsole.println("###########################");
		ResultConsole.println("|| ROOT STRUCTURE ACTIVATION RESULT:");
		ResultConsole.println("|| ============>> "+SuccessCounter+"/"+TestCounter);
		ResultConsole.println("###########################");
		
		int[] result = {};
		return result;
	}
	
	public void networkTest() 
	{
		NVTesting_Network nvtester = new NVTesting_Network(Console, ResultConsole);
		int TestCounter = 0;
		int SuccessCounter = 0;
		TestCounter++;
		SuccessCounter+=nvtester.testScalarBackprop();
	}
	public void componentTest() 
	{
		int TestCounter =0;
		int SuccessCounter=0;
		NVTesting_Component tester  = new NVTesting_Component(Console, ResultConsole);
		TestCounter++;
		SuccessCounter+=tester.testFunctionComponent();
		
		TestCounter++;
		SuccessCounter+=tester.testPropertyHub();
	}
	
	public void InputFormattingTest() 
	{
		NVTesting_Format tester = new NVTesting_Format(Console, ResultConsole);	
		int TestCounter = 0;
		int SuccessCounter = 0;
		NVertex n1 = new NVertex_Root(1,true, false, 0, 0, 1);
		Console.println("TESTING -> NVertex_Root(size: 1):");
		TestCounter++;
		SuccessCounter+=tester.inputSizeTest(n1, 1);
		
		n1 = new NVertex_Root(1,true, false, 0, 0, 1);
		Console.println("TESTING -> NVertex_Root (size 1):");
		TestCounter++;
		SuccessCounter+=tester.inputSizeTest(n1, 1);
		
		int vertexSize = 3;
		n1 = new NVertex_Root(vertexSize,true, false, 0, 0, 1);
		Console.println("TESTING -> NVertex_Root(size: "+vertexSize+"):");
		TestCounter++;
		SuccessCounter+=tester.inputSizeTest(n1, 1);
		
		n1 = new NVertex_Root(vertexSize,true, false, 0, 0, 1);
		Console.println("TESTING -> NVertex_Root (size "+vertexSize+"):");
		TestCounter++;
		SuccessCounter+=tester.inputSizeTest(n1, 1);
		
		Console.println("###########################");
		Console.println("|| INPUT FORMAT TEST RESULT:");
		Console.println("|| ============>> "+SuccessCounter+"/"+TestCounter);
		Console.println("###########################");
		ResultConsole.println("###########################");
		ResultConsole.println("|| INPUT FORMAT TEST RESULT:");
		ResultConsole.println("|| ============>> "+SuccessCounter+"/"+TestCounter);
		ResultConsole.println("###########################");
	}
	
	public void weightStateTest() 
	{
		NVTesting_Format tester = new NVTesting_Format(Console, ResultConsole);	
		int TestCounter = 0;
		int SuccessCounter = 0;
		//-----------------------------------------------------------
		NVertex head = new NVertex_Root(3, false, false, 0, 0, 1);
					
		double[][][] expected = {{null},{null},{null}};
		
		//head.setWeight(W);
		SuccessCounter 
		+= tester.testWeightState(head, expected, "testing on NVertex_Neuron");
		TestCounter++;
		//-----------------------------------------------------------
		head = new NVertex_Root(3, false, false, 0, 0, 1);
		
		double[][][] W = {{{1,-8,3}},{{-2,6,-1}},{{-6,4,-1}}};
		
		head.setWeight(W);
		SuccessCounter 
		+= tester.testWeightState(head, W, "testing on NVertex_Neuron");
		TestCounter++;
		//-----------------------------------------------------------
		head = new NVertex_Root(3, false, false, 0, 0, 1);
		NVertex src = new NVertex_Root(3, false, false, 1, 0, 1);
		head.connect(src); 
			
		head.setWeight(W);
		expected = W;
		SuccessCounter 
		+= tester.testWeightState(head, expected, "testing on NVertex_Neuron");
		TestCounter++;
		//-----------------------------------------------------------
			
		Console.println("###########################");
		Console.println("|| WEIGHT STATE TEST END RESULT:");
		Console.println("|| ============>> "+SuccessCounter+"/"+TestCounter);
		Console.println("###########################");
		
		ResultConsole.println("###########################");
		ResultConsole.println("|| WEIGHT STATE TEST END RESULT:");
		ResultConsole.println("|| ============>> "+SuccessCounter+"/"+TestCounter);
		ResultConsole.println("###########################");
	}
	
	public int[] advancedActivationTest()
	{
		NVTesting_Activation tester = new NVTesting_Activation(Console, ResultConsole);
		int TestCounter = 0;
		int SuccessCounter = 0;
		//-----------------------------------------------------------
		NVertex head = new NVertex_Root(3,true, false, 0, 0, 1);
		//Value size 1:
		double[][] I = {{4},{4},{4}};	
		double[]   expected = {4, 4, 4};
		SuccessCounter += tester.testNode_VectorActivation(head, I, expected, "{{4},{4},{4}} to {4,4,4} activation on single vertex");	
		TestCounter++;
		//-----------------------------------------------------------
		head = new NVertex_Root(3, true, false, 0, 0, 1);
		I[0][0] = -4; I[1][0] = -4; I[2][0] = -4;
		expected[0] = -0.04; expected[1] = -0.04; expected[2] = -0.04; 
		SuccessCounter += tester.testNode_VectorActivation(head, I, expected, "{{-4},{-4},{-4}} to {-0.04,-0.04,-0.04} activation on single vertex");	
		TestCounter++;
		
		//-----------------------------------------------------------
		head = new NVertex_Root(3, false, false, 0, 0, 1);
		NVertex src = new NVertex_Root(3, true, false, -1, 0, 1);
		double[][][] W = {{{1,-8,3}},{{-2,6,-1}},{{-6,4,-1}}};
		NVertex[] structure = {head, src};
		
		I[0][0] = -19; I[1][0] = 6; I[2][0] = 2;
		src.setInput(I);
		src.asExecutable().loadLatest();
		src.asExecutable().forward();
		src.asExecutable().loadLatest();

		head.connect(src);
		head.setWeight(W);
		expected[0] = -0.4219; expected[1] = 34.38; expected[2] = 23.14; 

		SuccessCounter 
		+= tester.testGraph_VectorActivation(structure, head, expected, "");
		TestCounter++;
		//-----------------------------------------------------------
		head = new NVertex_Root(3, false, false, 0, 0, 1);
		src = new NVertex_Root(3, true, false, -1, 0, 1);
		//double[][][] W = {{{1,-8,3}},{{-2,6,-1}},{{-6,4,-1}}};
		structure[0] = head; 
		structure[1] = src;
		
		I[0][0] = -19; I[1][0] = 6; I[2][0] = 2;
		src.setInput(I);
		src.asExecutable().loadLatest();
		src.asExecutable().forward();
		
		head.connect(src);
		head.setWeight(W);
		expected[0] = -0.4219; expected[1] = 34.38; expected[2] = 23.14; 

		SuccessCounter 
		+= tester.testGraph_VectorActivation(structure, head, expected, "");
		TestCounter++;
		//-----------------------------------------------------------
		//String fail = null; fail.hashCode();
		// To do:   vertex -convection-> vertex2
		Console.println("###########################");
		Console.println("|| ADVANCED ACTIVATION RESULT:");
		Console.println("|| ============>> "+SuccessCounter+"/"+TestCounter);
		Console.println("###########################");
		
		ResultConsole.println("###########################");
		ResultConsole.println("|| ADVANCED ACTIVATION RESULT:");
		ResultConsole.println("|| ============>> "+SuccessCounter+"/"+TestCounter);
		ResultConsole.println("###########################");
		
		int[] result = {SuccessCounter, TestCounter};
		return result;	
	}
	
	public int[] BaseFunctionsActivationTest()
	{
		NVTesting_Activation tester = new NVTesting_Activation(Console, ResultConsole);
		int TestCounter = 0;
		int SuccessCounter = 0;
		
		NVertex n1 = new NVertex_Root(1,true, false, 0, 0, 1);
		
		//Value size 1:
		double[] I = {4};	
		SuccessCounter += tester.testNode_ScalarActivation(n1, I, 4);	
		TestCounter++;
		
		I[0] = -2;	
		SuccessCounter += tester.testNode_ScalarActivation(n1, I, -0.02);
		TestCounter++;
		
		I[0] = -0.02;	
		SuccessCounter += tester.testNode_ScalarActivation(n1, I, -0.0002);
		TestCounter++;
		
		n1 = new NVertex_Root(1,true, false, 0, 5, 1);
		
		I[0] = 3;	
		SuccessCounter += tester.testNode_ScalarActivation(n1, I, 3);
		TestCounter++;
		
		I[0] = -3;	
		SuccessCounter += tester.testNode_ScalarActivation(n1, I, -3);
		TestCounter++;
		
		//Value size 2:
		//=====================================
		n1 = new NVertex_Root(1,true, false, 0, 5, 1);
		n1.addInput(0);
		I = new double[2]; I[0]=2; I[1]=3;
		SuccessCounter += tester.testNode_ScalarActivation(n1, I, 6);//test
		TestCounter++;
		
		n1 = new NVertex_Root(1,true, false, 0, 5, 1);
		n1.addInput(0);
		I = new double[2]; I[0]=4; I[1]=3;
		SuccessCounter += tester.testNode_ScalarActivation(n1, I, 12);//test
		TestCounter++;
		
		n1 = new NVertex_Root(1,true, false, 0, 5, 1);
		n1.addInput(0);
		I = new double[2]; I[0]=4; I[1]=3;
		SuccessCounter += tester.testNode_ScalarActivation(n1, I, 12);//test
		TestCounter++;
		
		//Value size 3:
		//====================================
		n1 = new NVertex_Root(1,true, false, 0, 5, 1);
		n1.addInput(0);
		n1.addInput(0);
		I = new double[3]; I[0]=2; I[1]=3; I[2]=4;
		SuccessCounter += tester.testNode_ScalarActivation(n1, I, 24);//test
		TestCounter++;
		
		n1 = new NVertex_Root(1,true, false, 0, 3, 1);
		n1.addInput(0);
		n1.addInput(0);
		I = new double[3]; I[0]=2; I[1]=3; I[2]=4;
		SuccessCounter += tester.testNode_ScalarActivation(n1, I, 576);//test
		TestCounter++;
		
		n1 = new NVertex_Root(1,true, false, 0, 4, 1);
		n1.addInput(0);
		n1.addInput(0);
		I = new double[3]; I[0]=2; I[1]=3; I[2]=4;
		SuccessCounter += tester.testNode_ScalarActivation(n1, I, 26.054189745157966);//test
		TestCounter++;
		
		n1 = new NVertex_Root(1,true, false, 0, 4, 1);
		n1.addInput(0);
		n1.addInput(0);
		I = new double[3]; I[0]=2; I[1]=-3; I[2]=4;
		SuccessCounter += tester.testNode_ScalarActivation(n1, I, 0.4152428423818953);//test
		TestCounter++;
		
		//================================
		I = new double[1];
		for(int Fi=0; Fi<8; Fi++) 
		{
			Console.println("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
			n1 = new NVertex_Root(1,true, false, 0, Fi, 1);
			
			double a = 0;
			switch(Fi) 
			{
				case 0: a=3;break;
				case 1: a=0.9525741268224331;break;
				case 2: a=0.9486832980505138;break;
				case 3: a=9;break;
				case 4: a=3.048587351573742;break;
				case 5: a=3;break;
				case 6: a=1.2340980408667962E-4;break;
				case 7: a=0.9486832980505138; break;//Default function: tanh
				default: a=0;
			}
			I[0] = 3;	
			SuccessCounter += tester.testNode_ScalarActivation(n1, I, a);
			TestCounter++;
			Console.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
		}
		
		Console.println("###########################");
		Console.println("|| NVertex_Neuron RESULT:");
		Console.println("|| ============>> "+SuccessCounter+"/"+TestCounter);
		Console.println("###########################");
		
		ResultConsole.println("###########################");
		ResultConsole.println("|| NVertex_Neuron RESULT:");
		ResultConsole.println("|| ============>> "+SuccessCounter+"/"+TestCounter);
		ResultConsole.println("###########################");
		
		Console.println("NVertex_Root Activation tests:");
		
		n1 = new NVertex_Root(1,true, false, 0, 0, 1);
		
		//Value size 1:
		I[0] = 4;	
		SuccessCounter += tester.testNode_ScalarActivation(n1, I, 4);	
		TestCounter++;
		
		I[0] = -2;	
		SuccessCounter += tester.testNode_ScalarActivation(n1, I, -0.02);
		TestCounter++;
		
		I[0] = -0.02;	
		SuccessCounter += tester.testNode_ScalarActivation(n1, I, -0.0002);
		TestCounter++;
		
		n1 = new NVertex_Root(1,true, false, 0, 5, 1);
		
		I[0] = 3;	
		SuccessCounter += tester.testNode_ScalarActivation(n1, I, 3);
		TestCounter++;
		
		I[0] = -3;	
		SuccessCounter += tester.testNode_ScalarActivation(n1, I, -3);
		TestCounter++;
		
		//Value size 2:
		//=====================================
		n1 = new NVertex_Root(1,true, false, 0, 5, 1);
		n1.addInput(0);
		I = new double[2]; I[0]=2; I[1]=3;
		SuccessCounter += tester.testNode_ScalarActivation(n1, I, 6);//test
		TestCounter++;
		
		n1 = new NVertex_Root(1,true, false, 0, 5, 1);
		n1.addInput(0);
		I = new double[2]; I[0]=4; I[1]=3;
		SuccessCounter += tester.testNode_ScalarActivation(n1, I, 12);//test
		TestCounter++;
		
		n1 = new NVertex_Root(1,true, false, 0, 5, 1);
		n1.addInput(0);
		I = new double[2]; I[0]=4; I[1]=4;
		SuccessCounter += tester.testNode_ScalarActivation(n1, I, 16);//test
		TestCounter++;
		
		//Value size 3:
		//====================================
		n1 = new NVertex_Root(1,true, false, 0, 5, 1);
		n1.addInput(0);
		n1.addInput(0);
		I = new double[3]; I[0]=2; I[1]=3; I[2]=4;
		SuccessCounter += tester.testNode_ScalarActivation(n1, I, 24);//test
		TestCounter++;
		
		n1 = new NVertex_Root(1,true, false, 0, 3, 1);
		n1.addInput(0);
		n1.addInput(0);
		I = new double[3]; I[0]=2; I[1]=3; I[2]=4;
		SuccessCounter += tester.testNode_ScalarActivation(n1, I, 576);//test
		TestCounter++;
		
		n1 = new NVertex_Root(1,true, false, 0, 4, 1);
		n1.addInput(0);
		n1.addInput(0);
		I = new double[3]; I[0]=2; I[1]=3; I[2]=4;
		SuccessCounter += tester.testNode_ScalarActivation(n1, I, 26.054189745157966);//test
		TestCounter++;
		
		n1 = new NVertex_Root(1,true, false, 0, 4, 1);
		n1.addInput(0);
		n1.addInput(0);
		I = new double[3]; I[0]=2; I[1]=-3; I[2]=4;
		SuccessCounter += tester.testNode_ScalarActivation(n1, I, 0.4152428423818953);//test
		TestCounter++;
		
		//================================
		I = new double[1];
		for(int Fi=0; Fi<8; Fi++) {
			Console.println("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
			n1 = new NVertex_Root(1,true, false, 0, Fi, 1);
			
			double a = 0;
			switch(Fi) 
			{
				case 0: a=3;break;
				case 1: a=0.9525741268224331;break;
				case 2: a=0.9486832980505138;break;
				case 3: a=9;break;
				case 4: a=3.048587351573742;break;
				case 5: a=3;break;
				case 6: a=1.2340980408667962E-4;break;
				case 7: a=0.9486832980505138; break;//Default function: tanh
				default: a=0;
			}
			I[0] = 3;	
			SuccessCounter += tester.testNode_ScalarActivation(n1, I, a);
			TestCounter++;
			Console.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
		}
		
		Console.println("###########################");
		Console.println("|| ACTIVATION END RESULT:");
		Console.println("|| ============>> "+SuccessCounter+"/"+TestCounter);
		Console.println("###########################");
		
		ResultConsole.println("###########################");
		ResultConsole.println("|| ACTIVATION END RESULT:");
		ResultConsole.println("|| ============>> "+SuccessCounter+"/"+TestCounter);
		ResultConsole.println("###########################");
		
		int[] result = {SuccessCounter, TestCounter};
		return result;	
	}
	

	//TODO: Put implementation innto format tester!...
	public void testCoreTypes() 
	{
		int SuccessCounter = 0;
		int TestCounter = 0;
		
		NVertex n 		= new NVertex_Root(1,false, false, 0, 0, 1);
		NVertex superUnit = new NVertex_Root(1,false, false, 0, 0, 1);
		
		printPropertiesOf(n);
		
		Console.println("Turn into basic core node:");

		NVertex.State.turnIntoBasicChild(n, superUnit);
		
		printPropertiesOf(n);
		TestCounter++;
		if(n.is(NVertex.BasicRoot)) 
		{
			Console.println("-> (unit.is(NVertex.BasicRoot)==true) -> test successful");
			SuccessCounter++;
		}
		else
		{
			Console.println("-> (unit.is(NVertex.BasicRoot)==False)! :( test failed");
			
		}
		TestCounter++;
		if(n.is(NVertex.Child)) 
		{
			Console.println("-> (unit.is(NVertex.Child)==true) -> test successful");
			SuccessCounter++;
		}
		else
		{
			Console.println("-> (unit.is(NVertex.Child)==false)! :( test failed");
			
		}
		
		//n.turnIntoSuperRootNode();
		NVertex.State.turnIntoParentRoot(n);
		printPropertiesOf(n);
		TestCounter++;
		if(n.is(NVertex.MotherRoot)) 
		{
			Console.println("-> (unit.is(NVertex.MotherRoot)==true) -> test successful");
			SuccessCounter++;
		}
		else
		{
			Console.println("-> (unit.is(NVertex.MotherRoot)==false)! :( test failed");
		}
	    Console.println("###########################");
	    Console.println("|| TYPE CONFIG TEST RESULT:");
		Console.println("|| ============>> "+SuccessCounter+"/"+TestCounter);
		Console.println("###########################");
		ResultConsole.println("###########################");
		ResultConsole.println("|| TYPE CONFIG TEST RESULT:");
		ResultConsole.println("|| ============>> "+SuccessCounter+"/"+TestCounter);
		ResultConsole.println("###########################");
	}
	
	
	public void printPropertiesOf(NVertex node) 
	{
		Console.println("\nNeuron diagnosis:");
		Console.println("==================");
		Console.println(node.toString());
		Console.println("==================");
	}
	
	
	
	
	
	
	
	
}
