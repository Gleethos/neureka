package neureka.unit;

import neureka.utility.NMessageFrame;

public class NVTesting_Component extends NVTesting{
	
	NVTesting_Component(NMessageFrame console, NMessageFrame resultConsole)
	{
		super(console, resultConsole);
	}
	
	public int testPropertyHub() {

		return 0;
	}
	
	public int testFunctionComponent() 
	{	
		NVTesting_Function tester = new NVTesting_Function(Console, ResultConsole);
		int TestCounter = 0;
		int SuccessCounter = 0;
		
		//EXPRESSION TESTING:
		TestCounter++;
		SuccessCounter += tester.testExpression("sum(ij)", "sum(I[j])", "");
		Console.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");
		
		TestCounter++;
		SuccessCounter += tester.testExpression("sum(1*(4-2/ij))", "sum(1.0*(4.0-(2.0/I[j])))", "");
		Console.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");
		
		TestCounter++;
		SuccessCounter += tester.testExpression("quadratic(ligmoid(Ij))", "quad(lig(I[j]))", "");
		Console.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");
		
		TestCounter++;
		SuccessCounter += tester.testExpression("softplus(I[3]^(3/i1)/sum(Ij^2)-23+I0/i1)", "lig((((I[3]^(3.0/I[1]))/sum(I[j]^2.0))-23.0)+(I[0]/I[1]))", "");
		Console.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");
		
		TestCounter++;
		SuccessCounter += tester.testExpression("1+3+5-23+I0*45/(345-651^I3-6)", "(1.0+3.0+(5.0-23.0)+(I[0]*(45.0/(345.0-(651.0^I[3])-6.0))))", "");
		Console.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");
		
		TestCounter++;
		SuccessCounter += tester.testExpression("sin(23*i1)-cos(i0^0.3)+tanh(23)", "((sin(23.0*I[1])-cos(I[0]^0.3))+tanh(23.0))", "");
		Console.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");

		TestCounter++;
		SuccessCounter += tester.testExpression("2*3/2-1", "((2.0*(3.0/2.0))-1.0)", "");
		Console.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");

		//ACTIVATION TESTING:
		TestCounter++;
		double[] input1 = {};
		SuccessCounter += tester.testActivation("6/2*(1+2)", input1, 9, "");
		Console.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");

		TestCounter++;
		input1 = new double[]{2,3.2,6};
		SuccessCounter += tester.testActivation("sum(Ij)", input1, 11.2, "");
		Console.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");
		
		TestCounter++;
		double[] input2 = {0.5,0.5,100};
		SuccessCounter += tester.testActivation("prod(Ij)", input2, 25, "");
		Console.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");
		
		TestCounter++;
		double[] input3 = {0.5,0.5,10};
		SuccessCounter += tester.testActivation("prod(prod(Ij))", input3, (2.5*2.5*2.5), "");
		Console.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");
		
		TestCounter++;
		double[] input4 = {5,4,3,12};//12/4-5+2+3
		SuccessCounter += tester.testActivation("I3/i[1]-I0+2+i2", input4, (3), "");
		Console.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");
		
		TestCounter++;
		double[] input5 = {-4,-2,6,-3,-8};//-3*-2/(-8--4-2)
		SuccessCounter += tester.testActivation("i3*i1/(i4-i0-2)-sig(0)+tanh(0)", input5, (-1.5), "");
		Console.println("||==>> "+SuccessCounter+"/"+TestCounter+"\n");
		
		Console.println("###########################");
		Console.println("|| FUNCTION END RESULT:");
		Console.println("|| ============>> "+SuccessCounter+"/"+TestCounter);
		Console.println("###########################");
		
		ResultConsole.println("###########################");
		ResultConsole.println("|| FUNCTION END RESULT:");
		ResultConsole.println("|| ============>> "+SuccessCounter+"/"+TestCounter);
		ResultConsole.println("###########################");
		
		
		
		
		return 0;
	}
	
	public int testMemoryComponent() {
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		return 0;
	}
	
	
	
	
	
	
	
	
	
	
}
