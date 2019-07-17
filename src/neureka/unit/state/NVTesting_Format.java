package neureka.unit.state;

import neureka.main.core.NVertex;
import neureka.utility.NMessageFrame;

public class NVTesting_Format {

	NMessageFrame Console;
	NMessageFrame ResultConsole;
	
	public NVTesting_Format(NMessageFrame console, NMessageFrame resultConsole){
		Console=console;
		ResultConsole=resultConsole;
	}
	
	public int testWeightState(NVertex head, double[][][] expected, String description) 
	{
		String bar = "[I]";
		Console.println("   Weight state test: "+description);
		Console.println("[O][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=]=>");
		Console.println(bar+"  ");
		Console.println(bar+"  Vertex: "+head.getFunction().toString());
		Console.println(bar+"====================================================");
		
		boolean success = true;
		
		double[][][] weight = head.getWeight();
		
		if(expected==null && weight!=null || expected!=null && weight==null)
		{
			if(expected==null)
			{
				Console.println(bar+"  (expected==null && weight!=null):=> test failed!");
				success = false;
			}
			else
			{
				Console.println(bar+"  (expected!=null && weight==null):=> test failed!");
				success = false;
			}
		}
		else
		{
			if(expected==null)
			{
				Console.println(bar+"  (expected==null && weight==null):=> equal!");
			}
			else
			{
				Console.println(bar+"  (expected!=null && weight!=null):");
				for(int Vi=0; Vi<expected.length && Vi<weight.length; Vi++) 
				{
					if(expected[Vi]==null && weight[Vi]!=null || expected[Vi]!=null && weight[Vi]==null)
					{
						if(expected[Vi]==null)
						{
							Console.println(bar+"  ('Vi':"+Vi+")|: (expected[Vi]==null && weight[Vi]!=null):=> test failed!");
							success = false;
						}
						else 
						{
							Console.println(bar+"  ('Vi':"+Vi+")|: (expected[Vi]!=null && weight==null):=> test failed!");
							success = false;
						}
					}
					else
					{
						if(expected[Vi]==null && weight[Vi]==null) 
						{
							Console.println(bar+"  ('Vi':"+Vi+")|: (expected[Vi]==null && weight==null):=> equal!");
						}
						else 
						{
							for(int Ii=0; Ii<expected[Vi].length && Ii<weight[Vi].length; Ii++) 
							{
								Console.println(bar+"  ('Vi':"+Vi+")|:('Ii':"+Ii+")|:");
								
								if(expected[Vi][Ii]==null && weight[Vi][Ii]!=null || expected[Vi][Ii]!=null && weight[Vi][Ii]==null)
								{
									if(expected[Vi][Ii]==null)
									{
										Console.println(bar+"  ('Vi':"+Vi+")|:('Ii':"+Ii+")|: (expected[Vi][Ii]==null && weight[Vi][Ii]!=null):=> test failed!");
										success=false;
									}
									else 
									{
										Console.println(bar+"  ('Vi':"+Vi+")|:('Ii':"+Ii+")|: (expected[Vi][Ii]!=null && weight[Vi][Ii]==null):=> test failed!");
										success=false;
									}
								}
								else
								{
									if(expected[Vi][Ii]==null && weight[Vi][Ii]==null) 
									{
										Console.println(bar+"  ('Vi':"+Vi+")|:('Ii':"+Ii+")|: (expected[Vi][Ii]==null && weight[Vi][Ii]==null):=> equal!");
									}
									else
									{
										for(int Wi=0; Wi<expected[Vi][Ii].length && Wi<weight[Vi][Ii].length; Wi++) 
										{
											String s = expected[Vi][Ii][Wi]+" =?= "+weight[Vi][Ii][Wi];
											Console.println(bar+"  ('Vi':"+Vi+")|:('Ii':"+Ii+")|:('Wi':"+Wi+")|: ("+s+"):");
											if(expected[Vi][Ii][Wi]!=weight[Vi][Ii][Wi]) {s="test failed!";}
											else {s="equal!";}
											Console.println(bar+"  ('Vi':"+Vi+")|:('Ii':"+Ii+")|:('Wi':"+Wi+")|: => "+s+"");
										}
										if(expected[Vi][Ii].length != weight[Vi][Ii].length)
										{
											Console.println(bar+"  ('Vi':"+Vi+")|:('Ii':"+Ii+")|: (expected[Vi][Ii].length != weight[Vi][Ii].length):=> test failed!");
											success = false;
										}
									}
								}
							}
							if(expected[Vi].length != weight[Vi].length)
							{
								Console.println(bar+"  ('Vi':"+Vi+")|: (expected[Vi].length != weight[Vi].length):=> test failed!");
								success = false;
							}
						}
					}
				}
			}
		}
		
		if(success) {return 1;}
		return 0;
	}
	public int inputSizeTest(NVertex core, int expectedSize) 
	{
		
		String bar = "[I]";
		Console.println("   input size test:");
		Console.println("[O][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=]=>");
		boolean success = true;
		for(int Vi=0; Vi<core.size(); Vi++) 
		{
			Console.println(bar+"  ");
			Console.println(bar+"  Value Size Test ('Vi':"+Vi+" of "+core.size()+"): "+core.getFunction().toString());
			Console.println(bar+"====================================================");
			Console.print(bar+"  Value size: "+core.inputSize()+"; Expected: "+expectedSize+"");
			if(core.inputSize()==expectedSize) 
			{
				Console.println(bar+"  Test succeeded! -> "+core.inputSize()+" == "+expectedSize);
			}
			else 
			{
				Console.println(bar+"  Test failed! -> "+core.inputSize()+" =|= "+expectedSize);	
				success=false;
			}
			Console.println(bar+"---------------------------------------------------");
			String message;
			//----------------------------------------------------------------------------
			if(core.getInput()[Vi].length==expectedSize) 
			{
				message = "SUCCESS => correctly sized!";
			}
			else 
			{
				success=false;
				message = "FAIL => array NOT correctly sized!";
			}
			Console.println(bar+"  Value array:           "+core.getInput()[Vi].length+message);
			//----------------------------------------------------------------------------
			if(core.getBias(Vi)==null)
			{
				success=false;
				message = "FAIL => Bias array of vertex element "+Vi+" is null!";
			}
			else if(core.getBias(Vi).length==expectedSize) 
			{
				message = "SUCCESS => correctly sized! ("+core.getBias(Vi).length+")";
			}
			else 
			{
				success=false;
				message = "FAIL => NOT correctly sized! ("+core.getBias(Vi)+")";
			}
			Console.println(bar+"   Bias array:            "+message);
			//----------------------------------------------------------------------------
			if(core.getWeight(Vi).length==expectedSize) 
			{
				message = "SUCCESS => correctly sized!";
			}
			else 
			{
				success=false;
				message = "FAIL => NOT correctly sized!";
			}	
			Console.println(bar+"  Matrix base array:     "+core.getWeight(Vi).length+message);
			//----------------------------------------------------------------------------
			if(core.getConnection().length==expectedSize) 
			{
				message = "SUCCESS => correctly sized!";
			}
			else 
			{
				success=false;
				message = "FAIL => NOT correctly sized!";
			}
			Console.println(bar+"  Connection base array: "+core.getConnection().length+message);
			//----------------------------------------------------------------------------
			Console.println(bar+"====================================================\n");
			
		}
		if(success==false) {return 0;}
		return 1;
	}
	
	
	
	
	
	
}
