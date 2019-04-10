package napi.main.core.opti;

import napi.main.core.NVNode;

public class NVOptimizer {

	private byte OptimizationID;
	//0 -> gradient + DeltaVelocity
	
	private double[][][][]  WeightGradientHistory;
	private double[][][]    ShiftGradientHistory;
	
	private boolean[][][][] WeightGradientSignHistory;
	private boolean[][][]   ShiftGradientSignHistory;
	
	private double[][][][]    WeightGradientOptimizationHistory;
	private double[][][]      ShiftGradientOptimizationHistory;
	//----------------------------------------	
	private double[][][]    WeightGradientVelocity;
	private double[][]      ShiftGradientVelocity;
	
	private double[][][]    WeightGradientDeltaVelocity;
	private double[][]      ShiftGradientDeltaVelocity;
	//----------------------------------------	
	private double[][][]    WeightGradientLearningRate;
	private double[][]	  ShiftGradientLearningRate;
			
	
	public NVOptimizer(int size, NVNode[][] connection, int optimizationID)
	{setUpNewOptimization(size, connection, optimizationID);}	
	
	public int getOptimizationID() {return OptimizationID;}
	//==============================================================================================================
	//==============================================================================================================
	public void update(int Ii, int Ni, boolean remove) {}
		
	public void setUpNewOptimization(int size, NVNode[][] connections, int optimizationID)
	{
		OptimizationID=(byte)optimizationID; if(connections==null) {System.out.println("Connection empty!!!");return;}
	
		System.out.println("NEW CONNECTION OPTIMIZATION!!!!!!");
	
		switch(optimizationID)
		{
		 case 0: //gradient + DeltaVelocity
			 instantiateGradientHistory(size, connections, 1);
			 instantiateGradientDeltaVelocity(size, connections);
			 nullSignHistory();nullGradientVelocity();nullGradientVelocity();nullGradientLearningRate(); nullGradientOptimizationHistory();
			 break;
		 case 1: //gradient * DeltaVelocity
			 instantiateGradientHistory(size, connections, 1);
			 instantiateGradientDeltaVelocity(size, connections);
			 nullSignHistory();nullGradientVelocity();nullGradientVelocity();nullGradientLearningRate(); nullGradientOptimizationHistory();
			 break;
		 case 2: //gradientVelocity * DeltaVelocity
			 instantiateGradientHistory(size, connections, 1);
			 instantiateGradientDeltaVelocity(size, connections);
			 instantiateGradientVelocity(size, connections);
			 nullSignHistory();nullGradientVelocity();nullGradientLearningRate(); nullGradientOptimizationHistory();
			 break;
		 case 3: //learningRate * gradient * DeltaVelocity
			 instantiateGradientHistory(size, connections, 1);
			 instantiateGradientOptimizationHistory(size, connections, 1);
			 instantiateGradientDeltaVelocity(size, connections);
			 instantiateGradientLearningRate(size, connections);
			 nullSignHistory();nullGradientVelocity();
			 break;
		 default: nullAll(); break;	 
		 }		
	}
	//Instantiation/Null Functions:
	//==============================================================================================================
	public void nullAll() 
	{nullGradientHistory();
	 nullSignHistory();	
	 nullGradientDeltaVelocity();
	 nullGradientVelocity();
	 nullGradientLearningRate();}
	//Histories:
	//----------------------------------------	
	private void nullGradientHistory() {WeightGradientHistory=null; ShiftGradientHistory=null;}	
	private void instantiateGradientHistory(int size, NVNode[][] connections, int historyLength)
	{
		WeightGradientHistory = new double[size][historyLength][connections.length][]; ShiftGradientHistory = new double[size][historyLength][connections.length];
		for(int Ui=0; Ui<size; Ui++)
		{ 
			for(int Hi=0; Hi<historyLength;Hi++) 
			{
				for(int Ii=0; Ii<connections.length; Ii++) 
				{
					if(connections[Ii]!=null) 
					{WeightGradientHistory[Ui][Hi][Ii] = new double[connections[Ii].length];}
				}
			}
		}
	}		
	//----------------------------------------
	private void nullSignHistory() {WeightGradientSignHistory=null; ShiftGradientSignHistory=null;}
	private void instantiateSignHistory(int size, NVNode[][] connections, int historyLength)
	{
		WeightGradientSignHistory = new boolean[size][historyLength][connections.length][]; 
		ShiftGradientSignHistory  = new boolean[size][historyLength][connections.length];
	
		for(int Ui=0; Ui<size; Ui++)
		{ 
			for(int Hi=0; Hi<historyLength;Hi++) 
			{
				for(int Ii=0; Ii<connections.length; Ii++) 
				{
					if(connections[Ii]!=null) {WeightGradientSignHistory[Ui][Hi][Ii] = new boolean[connections[Ii].length];}
				}
			}
		}
	}
	//----------------------------------------
	private void nullGradientOptimizationHistory() {WeightGradientOptimizationHistory=null; ShiftGradientOptimizationHistory=null; }	
	private void instantiateGradientOptimizationHistory(int size, NVNode[][] connections, int historyLength)
	{
		WeightGradientOptimizationHistory = new double[size][historyLength][connections.length][]; 
		ShiftGradientOptimizationHistory  = new double[size][historyLength][connections.length];
		for(int Ui=0; Ui<size; Ui++)
		{ 
			for(int Hi=0; Hi<historyLength;Hi++) 
			{
				for(int Ii=0; Ii<connections.length; Ii++) 
				{
					if(connections[Ii]!=null) {WeightGradientOptimizationHistory[Ui][Hi][Ii] = new double[connections[Ii].length];}
				}
			}
		}
	}
	
	//Velocities:
	//----------------------------------------	
	private void nullGradientDeltaVelocity() {WeightGradientDeltaVelocity=null; ShiftGradientDeltaVelocity=null;}	
	private void instantiateGradientDeltaVelocity(int size, NVNode[][] connections)
	{
		WeightGradientDeltaVelocity = new double[size][connections.length][]; 
		ShiftGradientDeltaVelocity = new double[size][connections.length];
		for(int Ui=0; Ui<size; Ui++)
		{ 
			for(int Ii=0; Ii<connections.length; Ii++) 
			{
				if(connections[Ii]!=null){WeightGradientDeltaVelocity[Ui][Ii] = new double[connections[Ii].length];}
			}
		}
	 }
	//----------------------------------------
	private void nullGradientVelocity() {WeightGradientVelocity=null; ShiftGradientVelocity=null;}	
	private void instantiateGradientVelocity(int size, NVNode[][] connections)
	{
		WeightGradientVelocity = new double[size][connections.length][]; ShiftGradientVelocity = new double[size][connections.length];
		for(int Ui=0; Ui<size; Ui++)
		{ 
			for(int Ii=0; Ii<connections.length; Ii++) 
			{
				if(connections[Ii]!=null) {WeightGradientVelocity[Ui][Ii] = new double[connections[Ii].length];}
			}
		}
	}
	//Learning Rate:
	//----------------------------------------	
	private void nullGradientLearningRate() {WeightGradientLearningRate=null; ShiftGradientLearningRate=null;}	
	private void instantiateGradientLearningRate(int size, NVNode[][] connections)
	{
		WeightGradientLearningRate = new double[size][connections.length][]; ShiftGradientLearningRate = new double[size][connections.length];
		for(int Ui=0; Ui<size; Ui++)
		{
			for(int Ii=0; Ii<connections.length; Ii++) 
			{
				if(connections[Ii]!=null) {WeightGradientLearningRate[Ui][Ii] = new double[connections[Ii].length];}ShiftGradientLearningRate[Ui][Ii]=1;
			}
			for(int Ii=0; Ii<connections.length; Ii++) 
			{
				if(connections[Ii]!=null) {for(int Ni=0; Ni<connections[Ii].length; Ni++){WeightGradientLearningRate[Ui][Ii][Ni]=1;}}
			}
		}
	}
	//==============================================================================================================
	
	//Public Interface:
	//==============================================================================================================
	public double getOptimized(double currentWeightGradient, int Vi,  int Iid, int Nid) 
	{switch(OptimizationID)
	 {
	 case 0: return gradientPlusDeltaVelocityOptimization(currentWeightGradient, Vi, Iid, Nid);
	 case 1: return gradientTimesDeltaVelocityOptimization(currentWeightGradient, Vi, Iid, Nid);
	 case 2: return gradientVelocityTimesDeltaVelocityOptimization(currentWeightGradient, Vi, Iid, Nid);
	 case 3: return learningRateTimesGradientTimesDeltaVelocityOptimization(currentWeightGradient, Vi, Iid, Nid);	  
	 case 4:
	 case 5: 
	 default: return currentWeightGradient;
	 }
	}
	//----------------------------------------	
	public double getOptimized(double currentBiasGradient, int Vi, int Iid)
	{switch(OptimizationID)
	 {		
	 case 0: return gradientPlusDeltaVelocityOptimization(currentBiasGradient, Vi, Iid);
	 case 1: return gradientTimesDeltaVelocityOptimization(currentBiasGradient, Vi, Iid);
	 case 2: return gradientVelocityTimesDeltaVelocityOptimization(currentBiasGradient, Vi, Iid);
	 case 3: return learningRateTimesGradientTimesDeltaVelocityOptimization(currentBiasGradient, Vi, Iid);	 
	 case 4:
	 case 5:		
	 }
	return currentBiasGradient;
	}
	//==============================================================================================================
	 
	//Subfunctions of Public Interface:
	//=================================
	// gradient + deltaV  
	//==============================================================================================================
	private double gradientPlusDeltaVelocityOptimization(double currentWeightGradient, int Ui, int Iid, int Nid)
	{
		if(WeightGradientHistory[Ui][Iid]!=null)
	    {
			double optimizedGradient; 
			WeightGradientDeltaVelocity[Ui][Iid][Nid]+=currentWeightGradient-WeightGradientHistory[Ui][0][Iid][Nid];		 
			optimizedGradient=currentWeightGradient+WeightGradientDeltaVelocity[Ui][Iid][Nid];		
			WeightGradientHistory[Ui]=updatedHistoryOf(WeightGradientHistory[Ui], Iid, Nid, currentWeightGradient);
			return optimizedGradient;	
	    }
	    else
	    {return currentWeightGradient;}
	 }
	//----------------------------------------
	private double gradientPlusDeltaVelocityOptimization(double currentWeightGradient, int Ui,  int Iid)
	{
		double optimizedGradient; ShiftGradientDeltaVelocity[Ui][Iid]+=currentWeightGradient-ShiftGradientHistory[Ui][Iid][0];		 
		optimizedGradient=currentWeightGradient+ShiftGradientDeltaVelocity[Ui][Iid];		
		ShiftGradientHistory[Ui]=updatedHistoryOf(ShiftGradientHistory[Ui], Iid, currentWeightGradient);
		return optimizedGradient;
    }
	//----------------------------------------	
	/*Arrays in use:
	 *
	 *GradientHistory / ShiftGradientHistory 
	 *GradientDeltaVelocity / ShiftGradientDeltaVelocity 
	 * 	 
	 */	
	// gradient * deltaV  
	//==============================================================================================================
	private double gradientTimesDeltaVelocityOptimization(double currentWeightGradient, int Ui,  int Iid, int Nid)
	{
		if(WeightGradientHistory[Ui][Iid]!=null)
		{
			double optimizedGradient; 
			WeightGradientDeltaVelocity[Ui][Iid][Nid]+=currentWeightGradient-WeightGradientHistory[Ui][0][Iid][Nid];
			
			if(WeightGradientDeltaVelocity[Ui][Iid][Nid]<0) 
			{
				optimizedGradient=currentWeightGradient*WeightGradientDeltaVelocity[Ui][Iid][Nid]*-1;
			}
			else 
			{
				optimizedGradient=currentWeightGradient*WeightGradientDeltaVelocity[Ui][Iid][Nid];
			}	
			
			WeightGradientHistory[Ui]=updatedHistoryOf(WeightGradientHistory[Ui], Iid, Nid, currentWeightGradient);
			return optimizedGradient;
		}
		else 
		{return currentWeightGradient;}
	}
	//----------------------------------------	
	private double gradientTimesDeltaVelocityOptimization(double currentBiasGradient, int Ui,  int Iid)
	{
		double optimizedGradient; 
	    ShiftGradientDeltaVelocity[Ui][Iid]+=currentBiasGradient-ShiftGradientHistory[Ui][0][Iid];	
	 
	    if(ShiftGradientDeltaVelocity[Ui][Iid]<0) 
        {
	    	optimizedGradient=currentBiasGradient*ShiftGradientDeltaVelocity[Ui][Iid]*-1;
	    }
        else 
        {
        	optimizedGradient=currentBiasGradient*ShiftGradientDeltaVelocity[Ui][Iid];	
        }	
	 	
	    ShiftGradientHistory[Ui]=updatedHistoryOf(ShiftGradientHistory[Ui], Iid, currentBiasGradient);
	    return optimizedGradient;
	}
	//----------------------------------------	
	/*Arrays in use:
	 *
	 *GradientHistory / ShiftGradientHistory 
	 *GradientDeltaVelocity / ShiftGradientDeltaVelocity 
	 * 	 
	 */
	// gradientV * deltaV  
	//==============================================================================================================
	private double gradientVelocityTimesDeltaVelocityOptimization(double currentWeightGradient, int Ui, int Iid, int Nid)
	{
		if(WeightGradientHistory[Iid]!=null)
		{
			double optimizedGradient; 
			WeightGradientDeltaVelocity[Ui][Iid][Nid]+=currentWeightGradient-WeightGradientHistory[Ui][0][Iid][Nid];	
			WeightGradientVelocity[Ui][Iid][Nid]+=currentWeightGradient;
			optimizedGradient=WeightGradientVelocity[Ui][Iid][Nid]*WeightGradientDeltaVelocity[Ui][Iid][Nid];		
			WeightGradientHistory[Ui]=updatedHistoryOf(WeightGradientHistory[Ui], Iid, Nid, currentWeightGradient);
			return optimizedGradient;
		}
		else
		{
			return currentWeightGradient;
		}
	}
	//----------------------------------------	
	private double gradientVelocityTimesDeltaVelocityOptimization(double currentBiasGradient, int Ui, int Iid)
	{
		double optimizedGradient; 
		ShiftGradientDeltaVelocity[Ui][Iid]+=currentBiasGradient-ShiftGradientHistory[Ui][0][Iid];	
		ShiftGradientVelocity[Ui][Iid]+=currentBiasGradient;
		optimizedGradient=ShiftGradientVelocity[Ui][Iid]*ShiftGradientDeltaVelocity[Ui][Iid];		
		ShiftGradientHistory[Ui]=updatedHistoryOf(ShiftGradientHistory[Ui], Iid, currentBiasGradient);
		return optimizedGradient;
	}
	//----------------------------------------	
	/*Arrays in use:
	 *
	 *GradientVelocity / ShiftGradientVelocity
	 *GradientHistory / ShiftGradientHistory 
	 *GradientDeltaVelocity / ShiftGradientDeltaVelocity 
	 */
	
	// learningR * gradient * deltaV  
	//==============================================================================================================	
	private double learningRateTimesGradientTimesDeltaVelocityOptimization(double currentWeightGradient, int Ui, int Iid, int Nid)
	{
		if(WeightGradientHistory[Iid]!=null)
		{
			double optimizedGradient=0;	 
			if(WeightGradientHistory[Ui][0][Iid][Nid]>0 && currentWeightGradient>0) 
			{//--------------------------------------------------------------------------------------------------------------
				if(WeightGradientHistory[Ui][0][Iid][Nid]>(currentWeightGradient)) {WeightGradientLearningRate[Ui][Iid][Nid]*=0.8;}
				if(WeightGradientHistory[Ui][0][Iid][Nid]<(currentWeightGradient)) {WeightGradientLearningRate[Ui][Iid][Nid]*=1.2;}	 
			}//--------------------------------------------------------------------------------------------------------------	 
			if(WeightGradientHistory[Ui][0][Iid][Nid]<0 && currentWeightGradient<0) 
			{//--------------------------------------------------------------------------------------------------------------
				if(WeightGradientHistory[Ui][0][Iid][Nid]>(currentWeightGradient)) {WeightGradientLearningRate[Ui][Iid][Nid]*=0.8;}
				if(WeightGradientHistory[Ui][0][Iid][Nid]<(currentWeightGradient)) {WeightGradientLearningRate[Ui][Iid][Nid]*=1.2;} 
			}//--------------------------------------------------------------------------------------------------------------	 	
			if(WeightGradientHistory[Ui][0][Iid][Nid]>0 && currentWeightGradient<0) 
			{//--------------------------------------------------------------------------------------------------------------
				if(WeightGradientHistory[Ui][0][Iid][Nid]>(currentWeightGradient*-1)) {WeightGradientLearningRate[Ui][Iid][Nid]*=0.8;}
				if(WeightGradientHistory[Ui][0][Iid][Nid]<(currentWeightGradient*-1)) {WeightGradientLearningRate[Ui][Iid][Nid]*=0.5;return WeightGradientOptimizationHistory[Ui][0][Iid][Nid]*=-1;}	 
			}//--------------------------------------------------------------------------------------------------------------	 
			if(WeightGradientHistory[Ui][0][Iid][Nid]<0 && currentWeightGradient>0) 
			{//--------------------------------------------------------------------------------------------------------------
				if((WeightGradientHistory[Ui][0][Iid][Nid]*-1)>currentWeightGradient) {WeightGradientLearningRate[Ui][Iid][Nid]*=0.8;}
				if((WeightGradientHistory[Ui][0][Iid][Nid]*-1)<currentWeightGradient) {WeightGradientLearningRate[Ui][Iid][Nid]*=0.5;return WeightGradientOptimizationHistory[Ui][0][Iid][Nid]*=-1;}	 		 
			}//--------------------------------------------------------------------------------------------------------------		
			WeightGradientDeltaVelocity[Ui][Iid][Nid]+=currentWeightGradient-WeightGradientHistory[Ui][0][Iid][Nid];
			//=============================================================================================
			if(WeightGradientDeltaVelocity[Ui][Iid][Nid]<0) 
			{
				optimizedGradient=currentWeightGradient*WeightGradientDeltaVelocity[Ui][Iid][Nid]*-1*WeightGradientLearningRate[Ui][Iid][Nid];
			}
			else 
			{
				optimizedGradient=currentWeightGradient*WeightGradientDeltaVelocity[Ui][Iid][Nid]*WeightGradientLearningRate[Ui][Iid][Nid];
			}	
			WeightGradientOptimizationHistory[Ui][0][Iid][Nid]=optimizedGradient;	
			WeightGradientHistory[Ui]=updatedHistoryOf(WeightGradientHistory[Ui], Iid, Nid, currentWeightGradient);
			return optimizedGradient;	
		}
		else
		{return currentWeightGradient;}
	}
	//----------------------------------------	
	private double learningRateTimesGradientTimesDeltaVelocityOptimization(double currentBiasGradient, int Ui, int Iid)
	{
		double optimizedGradient=0;	 
		if(ShiftGradientHistory[Ui][0][Iid]>0 && currentBiasGradient>0) 
		{
			if(ShiftGradientHistory[Ui][0][Iid]>(currentBiasGradient)) {ShiftGradientLearningRate[Ui][Iid]*=0.8;}
			if(ShiftGradientHistory[Ui][0][Iid]<(currentBiasGradient)) {ShiftGradientLearningRate[Ui][Iid]*=1.2;}	 
		}	 
		if(ShiftGradientHistory[Ui][0][Iid]<0 && currentBiasGradient<0) 
		{
			if(ShiftGradientHistory[Ui][0][Iid]>(currentBiasGradient)) {ShiftGradientLearningRate[Ui][Iid]*=0.8;}
			if(ShiftGradientHistory[Ui][0][Iid]<(currentBiasGradient)) {ShiftGradientLearningRate[Ui][Iid]*=1.2;} 
		}	 	
		if(ShiftGradientHistory[Ui][0][Iid]>0 && currentBiasGradient<0) 
		{
			if(ShiftGradientHistory[Ui][0][Iid]>(currentBiasGradient*-1)) {ShiftGradientLearningRate[Ui][Iid]*=0.8;}
			if(ShiftGradientHistory[Ui][0][Iid]<(currentBiasGradient*-1)) {ShiftGradientLearningRate[Ui][Iid]*=0.5;return ShiftGradientOptimizationHistory[Ui][0][Iid]*=-1;}	 
		}	 
		if(ShiftGradientHistory[Ui][0][Iid]<0 && currentBiasGradient>0) 
		{
			if((ShiftGradientHistory[Ui][0][Iid]*-1)>currentBiasGradient) {ShiftGradientLearningRate[Ui][Iid]*=0.8;}
			if((ShiftGradientHistory[Ui][0][Iid]*-1)<currentBiasGradient) {ShiftGradientLearningRate[Ui][Iid]*=0.5;return ShiftGradientOptimizationHistory[Ui][0][Iid]*=-1;}	 		 
		}		
		ShiftGradientDeltaVelocity[Ui][Iid]+=currentBiasGradient-ShiftGradientHistory[Ui][0][Iid];	
		
		if(ShiftGradientDeltaVelocity[Ui][Iid]<0) 
        {optimizedGradient=currentBiasGradient*ShiftGradientDeltaVelocity[Ui][Iid]*-1*ShiftGradientLearningRate[Ui][Iid];}
		else 
		{optimizedGradient=currentBiasGradient*ShiftGradientDeltaVelocity[Ui][Iid]*ShiftGradientLearningRate[Ui][Iid];	}
		
		ShiftGradientOptimizationHistory[Ui][0][Iid]=optimizedGradient;	
		ShiftGradientHistory[Ui]=updatedHistoryOf(ShiftGradientHistory[Ui], Iid, currentBiasGradient);
		System.out.println("Optimization: currentBiasGradient("+currentBiasGradient+") -> optimizedShiftGradient: "+optimizedGradient+" ...Also: ShiftLearningRate: "+ShiftGradientLearningRate[Ui][Iid]);
		return optimizedGradient;		
	}
	//----------------------------------------	
	/*Arrays in use:
	 *
	 *GradientLearningRate /ShiftGradientLearningRate
	 *GradientHistory / ShiftGradientHistory 
	 *GradientDeltaVelocity / ShiftGradientDeltaVelocity 
	 */
		
	//Sub-Subfunctions of Public Interface:
	//=====================================
	//History management:
	//========================================================================================
	private double[][][] updatedHistoryOf(double[][][] History, int Iid, int Nid, double latest)
	{
		for(int Hi=History.length-2; Hi>=0; Hi--) 
	    {
			History[Hi+1][Iid][Nid]=History[Hi][Iid][Nid];
	    }
		History[0][Iid][Nid]=latest;
		return History;
	}
	//----------------------------------------------------------------------------------------
	private double[][] updatedHistoryOf(double[][] History, int Iid, double latest)
	{
		for(int Hi=History.length-2; Hi>=0; Hi--) 
	    {
			History[Hi+1][Iid]=History[Hi][Iid];
	    }
		History[0][Iid]=latest;
		return History;
	}
	//----------------------------------------------------------------------------------------
	private double[] updatedHistoryOf(double[] History, double latest)
	{
		for(int Hi=History.length-2; Hi>=0; Hi--) 
	    {
			History[Hi+1]=History[Hi];
	    }
		History[0]=latest;
		return History;
	}
	//========================================================================================

}
