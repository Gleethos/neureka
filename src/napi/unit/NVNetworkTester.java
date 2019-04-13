
package napi.unit;

import java.math.BigInteger;
import java.util.ArrayList;

import napi.main.core.NVNode;
import napi.main.core.NVertex;
import napi.main.core.calc.NVFHead;
import napi.main.core.calc.NVFunction;
import napi.main.core.imp.NVertex_Root;
import napi.main.core.opti.NVOptimizer;
import napi.utility.NMessageFrame;

public class NVNetworkTester 
{
	ArrayList<NVertex> NodeList;
	NMessageFrame Console;
	NMessageFrame ResultConsole;
	
	NVNetworkTester(NMessageFrame console, NMessageFrame resultConsole)
	{
		Console=console;
		ResultConsole=resultConsole;
	}
	private void createNeuron()
	{
		createNeuron(false, false, 0);
	}
	private void createNeuron(boolean isFirst, boolean isLast, int functionID)
	{
		NVertex N = new NVertex_Root(1, isFirst, isLast, 0, 0);NodeList.add(N);
	}
	private void ForwardpropagateFor(int timeSteps)
	{
		for(int t=0; t<timeSteps; t++)
	     {startForwardPropagation();}
	}
	private void startForwardPropagation()
	{
		for(int Ni=0; Ni<NodeList.size(); Ni++) 
		{
			NVertex u = NodeList.get(Ni);

			u.asExecutable().loadState();
		}
		for(int Ni=0; Ni<NodeList.size(); Ni++) 
		{
			NVertex u = NodeList.get(Ni);
			u.asExecutable().forward();
		}
	}
	private void BackpropagateFor(int timeSteps)
	{
		for(int t=0; t<timeSteps; t++)
	    {
			startBackpropagation(new BigInteger(String.valueOf(t)));
		}
	}
	private void startBackpropagation(BigInteger Hi) 
	{
		for(int Ni=0; Ni<NodeList.size(); Ni++) 
		{
			NodeList.get(Ni).asExecutable().preBackward(Hi);
		}
		for(int Ni=0; Ni<NodeList.size(); Ni++) 
		{
			NodeList.get(Ni).asExecutable().Backward(Hi);
		}
	}
	public void connectNToNInput(NVertex neuronOne, NVertex neuronTwo) 
	{	
		neuronTwo.connect(neuronOne);
		neuronTwo.randomizeBiasAndWeight();	
		neuronTwo.displayCurrentState();
		NVNode[][] Connection = neuronTwo.getConnection();
		for(int Ii=0; Ii<Connection.length; Ii++)
		{
			neuronTwo.setBias(0, Ii, -3);
			for(int Ni=0; Ni<Connection[Ii].length; Ni++) 
			{
				System.out.println("Ii:"+Ii+"; Ni:"+Ni+";");
				neuronTwo.setWeight(0, Ii, Ni, 2.0);
			}
		}
	}
	
	
	public void Test()
	{
		NodeList = new ArrayList<NVertex>();
		NVertex[] Net = getTinyTestNet();
		NVertex N1 = Net[0]; NodeList.add(N1);	
		NVertex N2 = Net[1]; NodeList.add(N2);	
		NVertex N3 = Net[2]; NodeList.add(N3);
		NVertex N4 = Net[3]; NodeList.add(N4);	
		N4.setLast(false);
		for(int i=0; i<4; i++) 
		{
			ForwardpropagateFor(4); 
			BackpropagateFor(4);
		}
		startForwardPropagation();
		NodeList = new ArrayList<NVertex>();
	}
	
	
	public int testScalarBackprop()
	{
		NodeList = new ArrayList<NVertex>();
		NVertex[] Net = getTinyTestNet();
		NVertex N1 = Net[0]; 	NodeList.add(N1);	
		NVertex N2 = Net[1];	NodeList.add(N2);	
		NVertex N3 = Net[2];	NodeList.add(N3);
		NVertex N4 = Net[3];	NodeList.add(N4);
		
		NVertex N5 = new NVertex_Root(1, false, false, -1, 5);
		NodeList.add(N5);//N5 is loss function!
		
		N5.setConvergence(true);

		System.out.println("N4 function: "+N4.getFunction().expression());
		
		double optimum = 376;
		N4.setOptimum(0,optimum);
		//N5.setOptimum(0,0);
		
		N5.connect(N4);
		N5.setWeight(null);
		N5.setBias(null);
		
		NVFunction F = new NVFHead();
		F = F.newBuild("abs(I0)");
		N5.setFunction(F);
		
		NMessageFrame Frame = new NMessageFrame("Network test - backprop - Output");
		int[] counter = {0};
		NodeList.forEach
		(
			(n)->
			{
				Frame.println("N["+counter[0]+"]:"+n.toString()); counter[0]++;
			}
		);
		String bar = "[I]";
		Console.println("   Backprop test on tiny test net: ");
		Console.println("[O][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=]=>");
		Console.println(bar+"  ");
		Console.println(bar+"  Head: "+N4.getFunction().expression());
		Console.println(bar+"  Loss: "+N5.getFunction().expression());
		Console.println(bar+"====================================================");
		
		//for(int i=0; i<Net.length; i++) {Frame.print(Net[i].toString());}
		Frame.println("N4 activation:"+N4.getActivation(0));
		Frame.println("N5 activation:"+N5.getActivation(0));
		Frame.println("=================");
		
		Console.println(bar+"======================");
		Console.println(bar+"4*forward, 4*backward ...");		
		Console.println(bar+"======================");
		for(int i=0; i<100; i++) 
		{
			Frame.println("...4 x forward, ...4 x backward:");
			ForwardpropagateFor(4); 
			BackpropagateFor(4);
			Frame.println("Result:");
			Frame.println("N4 activation:"+N4.getActivation(0));
			Frame.println("N5 activation:"+N5.getActivation(0));
			//Frame.println("N5 data:"+N5.toString());
		}
		ForwardpropagateFor(4); 

		Console.println(bar+"=================");
		Console.println(bar+"N4 activation:"+N4.getActivation(0));
		Console.println(bar+"N5 activation:"+N5.getActivation(0));
		Console.println(bar+"=================");
		System.out.println(N4.toString());
		//startForwardPropagation();
		NodeList = new ArrayList<NVertex>();
		int result = 0;
		if((N4.getActivation(0)-optimum)<0.1 && N5.getActivation(0)<0.1) {
			Console.println(bar+"(N4.Activation(0)-optimum)<0.1  -=> success!");
			Console.println(bar+"N5.Activation(0)<0.1  -=> success!");
			result = 1;
		}
		else {
			Console.println(bar+"((N4.Activation(0)-optimum)<0.1 && N5.Activation(0)<0.1) == false  -=> test failed!");
			result = 0;
		}
		Console.println(bar+"====================================================\n");
		
		
		//String fail = null; fail.hashCode();
		return result;
	}
	
	
	
	private NVertex[] getTinyTestNet() 
	{
		NVertex[] N = new NVertex[4];
		
		N[0] = new NVertex_Root(1, true, false, 0, 5); 
		N[1] = new NVertex_Root(1, false, false, 1, 5);
		N[2] = new NVertex_Root(1, false, false, 2, 5);
		N[3] = new NVertex_Root(1, false, true, 3, 5);
		
		N[3].setConvergence(true);
		N[0].setAllInput(0,6);
		N[3].setOptimum(0, 15);
		System.out.println("N4 function: "+N[3].getFunction().expression());
	 
		System.out.println("Connecting N1 To N2:");
		connectNToNInput(N[0],N[1]);
		System.out.println("Connecting N1 To N3:");
		connectNToNInput(N[0],N[2]);
		System.out.println("Connecting N2 To N4:");
		connectNToNInput(N[1],N[3]);
		System.out.println("Connecting N3 To N4:");
		connectNToNInput(N[2],N[3]);
		/**			 /-->[1] --\
		 * 	  [0]===|			|==>[3]
		 * 			 \-->[2] --/
		 * */
		N[0].setBias(0, 0, -3);
		
		N[1].setBias(0, 0, 2);
		N[1].setWeight(0, 0, 0, -3.0);
	
		N[2].setBias(0, 0, -5);
		N[2].setWeight(0, 0, 0, 4.0);
		
		N[3].setBias(0, 0, 3.5);
		N[3].setWeight(0, 0, 0, 6.0);
		N[3].setBias(0, 1, -25);
		N[3].setWeight(0, 1, 0, 6.0);
		
		N[0].addModule(new NVOptimizer(N[0].size(),N[0].getConnection(),3));
		N[1].addModule(new NVOptimizer(N[1].size(),N[1].getConnection(),3));
		N[2].addModule(new NVOptimizer(N[2].size(),N[2].getConnection(),3));
		N[3].addModule(new NVOptimizer(N[3].size(),N[3].getConnection(),3));
		
		return N;
	}
	
	
	
}
