
public class NHistoryManagment {

	
	NHistoryManagment(){}	
	//======================
	//========================================================================================
	public double[][][] updatedHistoryOf(double[][][] History, int Iid, int Nid, double latest)
	{for(int Hi=History.length-2; Hi>=0; Hi--) 
	     {
	      History[Hi+1][Iid][Nid]=History[Hi][Iid][Nid];
	     }
	 History[0][Iid][Nid]=latest;
	 return History;
	}
	//========================================================================================
	public double[][] updatedHistoryOf(double[][] History, int Iid, double latest)
	{for(int Hi=History.length-2; Hi>=0; Hi--) 
	     {
	      History[Hi+1][Iid]=History[Hi][Iid];
	     }
	 History[0][Iid]=latest;
	 return History;
	}
	//----------------------------------------------------------------------------------------
	public double[][] updatedHistoryOf(double[][] History, double[] latest)
	{for(int Hi=History.length-2; Hi>=0; Hi--) 
	     {for(int Nid=0; Nid<History[Hi].length; Nid++) 
	          {History[Hi+1][Nid]=History[Hi][Nid];}
	     }
	
	for(int Nid=0; Nid<History[0].length; Nid++) 
        {History[0][Nid]=latest[Nid];}
	 
	 return History;
	}
	//========================================================================================
	public double[] updatedHistoryOf(double[] History, double latest)
	{for(int Hi=History.length-2; Hi>=0; Hi--) 
	     {
	      History[Hi+1]=History[Hi];
	     }
	 History[0]=latest;
	 return History;
	}
	//----------------------------------------------------------------------------------------
	public boolean[] updatedHistoryOf(boolean[] History, boolean latest)
	{for(int Hi=History.length-2; Hi>=0; Hi--) 
	     {
	      History[Hi+1]=History[Hi];
	     }
	 History[0]=latest;
	 return History;
	}
	//----------------------------------------------------------------------------------------
	public boolean[] extendedHistoryOf(boolean[] History, boolean latest)
	{boolean[] newHistory = new boolean[History.length+1];
	 newHistory[0]=latest;
	 for(int Hi=0; Hi<History.length; Hi++) 
	     {
	      newHistory[Hi+1]=History[Hi];
	     }
	 return newHistory;
	}
	//========================================================================================
	
		
	
}
