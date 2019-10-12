package gpu.wip;

import java.util.Random;

import com.aparapi.Kernel.Constant;

public final class NVCElement
{
	
	public static AdvancedNVConvector Root;
	private int row;
	//----------------------------------------------------
	public void setRoot(AdvancedNVConvector root) {Root=root;}
    //----------------------------------------------------
    public int ins() {return Anchor_AoBcomp_size;}
	public int rls() {return ins();}
	public int row() {return row;} 
	//-----------------------------------------------
	//################################################
	//private int[] AoBcomp;// Defines composition of matrix A over B by defining true/false periods over Bcomp!
	//private int[] Anchor_AoBcomp;// Defines size of matrix A vector for every input layer
	//private int[] Anchor_Arowlayer;// Defines row layer anchors
	//private float matA[];
	//private float matC[];
	//private float C[];
	//-------------------
	int AoBcomp_start=0;
	int Anchor_AoBcomp_start=0;
	int Anchor_Arowlayer_start=0;
	
	int AoBcomp_size;
	int Anchor_AoBcomp_size;
	//int Anchor_Arowlayer_size;
	
	int matA_size;
	int matC_size;
	int C_size;
	
	int matA_start=0;
	int matC_start=0;
	int C_start=0;  
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    public float[] matA()      {return Root.matA();}
    public float   matA(int i) {return Root.matA(matA_start+i);}//WARNING! : Not fit for multiple elements!
    public void    setMatA(float[] newA) {Root.setMatA(newA);}
  //-----------------------------------------------
    public float[] matC()      {return Root.matC();}
    public float   matC(int i) {return Root.matC(i);}
    public void    setMatC(float[] newC) {Root.setMatC(newC);}
    public void    setMatC(int i, float value) {Root.setMatC(i,value);}
  //-----------------------------------------------
    public float[] C()      {return Root.C();}
    public float   C(int i) {return Root.C(i);}
    public void	   setC(float[] newC) {Root.setC(newC);}
    public void    setC(int i, float value) {Root.setC(i,value);} 
  //-----------------------------------------------
    public int[] AoBcomp()      {return Root.AoBcomp();}
    public int   AoBcomp(int i) {return Root.AoBcomp(i);}
    
    public int[] Anchor_AoBcomp()      {return Root.Anchor_AoBcomp();}
    public int   Anchor_AoBcomp(int i) {return Root.Anchor_AoBcomp(i);}
    
    public int[] Anchor_Arowlayer()      {return Root.Anchor_Arowlayer();}
    public int   Anchor_Arowlayer(int i) {return Root.Anchor_Arowlayer(i);}
  //------------------------------------------------------------------
  
     
    
	public NVCElement(AdvancedNVConvector root, int row, int collom, int[] AoverBComposition, int[] Bcomposition) 
	{
		Root = root;
		AoBcomp_size = AoverBComposition.length;
		
		
		//Expanding array in root!
		//AoBcomp = AoverBComposition; 
		Root.addAoBcomp(0, AoverBComposition);
		
    	// Stores numbers representing active/inactive connection times for the Bcomp array (positive/negative)
    	// for every input dimension 
    	//==> Distribution is defined by the following anchor array:
    	//-------------------------------------------------------------------------------------------------------

		
		
		
    	//===================================================
    	//Calculating input size from A_toBcomp:
    	//===================================================
    	int comcount = 0;
    	int Insi = 0;
    	System.out.println("Bcomp.length: "+Bcomposition.length);
    	for(int i=0; i<AoBcomp_size; i++) 
    	{
    		System.out.println("AoBcomp: ["+i+"]=>["+Root.AoBcomp(i)+"]; comcount: "+comcount);
    		if(comcount==Bcomposition.length) 
    		{comcount=0;Insi++;}
    		comcount+=Math.abs(Root.AoBcomp(i));
    	}
    	int anchoring=0;
    	Root.addAnchor_AoBcomp(0, new int[Insi+1]);
    	// expanding array in root!
    	
    	Anchor_AoBcomp_size = Insi+1;
    	Insi=0;comcount=0;
    	for(int i=0; i<AoBcomp_size; i++) 
    	{
    		if(comcount==Bcomposition.length) 
    		{comcount=0;Root.setAnchor_AoBcomp(Insi,anchoring);Insi++;}
    		comcount+=Math.abs(Root.AoBcomp(i));
    		anchoring ++;
    	}
    	Root.setAnchor_AoBcomp(Insi,anchoring);
    	//===================================================//The above code works!
    	for(int i=0; i<Anchor_AoBcomp_size; i++) {System.out.println("Anchor_AoBcomp: [->"+Root.Anchor_AoBcomp(i)+"]");}
    	//===================================================

    	//-----------------------------------------
    	Root.addAnchor_Arowlayer(0, new int[Anchor_AoBcomp_size]);// ==> ROW LAYER ANCHOR
    	//Defines anchor points for every input on one row layer.
    	//-----------------------------------------
    	
    	int rowlayer_inputanchoring=0;
    	//=======================================
    	for(Insi=0; Insi<Anchor_AoBcomp_size; Insi++)// ? : ACA -> size == input size !!
    	{
    		int AoBcomp_anch_p;
    		System.out.println("\nNew insi index: "+Insi);
    		System.out.println("=========================");
    		//------------------
    		if(Insi!=0) // Setting anchorpoints:
    		{AoBcomp_anch_p=Root.Anchor_AoBcomp(Insi-1);}else{AoBcomp_anch_p=0;}
    		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    		System.out.println("AoBcomp_anch_p set to: "+AoBcomp_anch_p);
    		
    		//-------------------------
    		//Sub array iteration (connection phase discriptor)
    		int Bcomp_i=0;
    		while(AoBcomp_anch_p<(Root.Anchor_AoBcomp(Insi)))// p iterating to the next subarray! 
    		{
    			System.out.println("\n==>> While(AoBcomp_anch_p<(Root.Anchor_AoBcomp(Insi))):("+AoBcomp_anch_p+"<"+(Root.Anchor_AoBcomp(Insi))+")");
    			//################################// This is within one row where Ii & Ri is given!
    			if(0<Root.AoBcomp(AoBcomp_anch_p))// 
    			{//::::::::::::::::::::::
    				int anchor = Bcomp_i+Root.AoBcomp(AoBcomp_anch_p);
    				System.out.println("==>>==>> anchor: "+anchor);
    				while(Bcomp_i<anchor) 
    				{
    					rowlayer_inputanchoring += Bcomposition[Bcomp_i];
    					Bcomp_i++;
    				}
    			}//::::::::::::::::::::::
    			else//======================>>>> A_comp contains within itself A configuration using B config within Bcomp
    			{//::::::::::::::::::::::
    				System.out.println("==>>==>> Bcomp_i+=AoBcomp[A_comp_anch]*-1  <->  "+Bcomp_i+"+="+Root.AoBcomp(AoBcomp_anch_p)*-1+"");
    				Bcomp_i+=Root.AoBcomp(AoBcomp_anch_p)*-1;
    			}//::::::::::::::::::::::
    			//################################
    			
    			AoBcomp_anch_p++;	
    		}
    		//-------------------------
    		System.out.println("Anchor_Arowlayer[Insi]=rowlayer_inputanchoring <-> "+Root.Anchor_Arowlayer(Insi)+"="+rowlayer_inputanchoring);
    		Root.setAnchor_Arowlayer(Insi,rowlayer_inputanchoring);//This works!
    	}
    	//=======================================// Yasss!!
    	
    	int A_RowLayerSize = Root.Anchor_Arowlayer(Root.Anchor_Arowlayer().length-1);
    	//=======================================
    	System.out.println("A_RowLayerSize: "+A_RowLayerSize);
    	
    	
        this.row = row;
        //---------------------------------------------------
        Root.setMatA(new float [A_RowLayerSize *  row]);
        Root.setMatC(new float [row * collom * Anchor_AoBcomp_size]);
        //---------------------------------------------------
        Root.setC(new float[row * collom * Anchor_AoBcomp_size]);
        //---------------------------------------------------
        
        int counter=0;
        //##################################################################################################################################################################################>
	    //===================================================================================================================================================================================>
	    //##################################################################################################################################################################################>
        for(Insi=0; Insi<Anchor_AoBcomp_size; Insi++) 
        {
        	//Here matrix A is initialized with random number
        	for(int Rowi = 0; Rowi < row; Rowi++ )
        	{
        		int anchbase;
    	    	if(Insi==0) {anchbase=0;}else {anchbase=Root.Anchor_Arowlayer(Insi-1);}
    	    	int inrowsize = Root.Anchor_Arowlayer(Insi)-anchbase;
    			//---------------
    			int AoBcomp_anch_p;// Setting anchor points:
    			if(Insi!=0) {AoBcomp_anch_p=Root.Anchor_AoBcomp(Insi-1);}else {AoBcomp_anch_p=0;}
    			//-------------------------
    			int Bcomp_i=0;
    			int Acomi=0;
    			int Bcomi=0;
    			//Sub array iteration (connection phase discriptor)
    			while(AoBcomp_anch_p<(Root.Anchor_AoBcomp(Insi))) // p iterating to the next subarray!
    			{
    				//################################// This is within one row where Ii & Ri is given!
    				if(0<Root.AoBcomp(AoBcomp_anch_p)) 
    				{//::::::::::::::::::::::
    					int anchor = Bcomp_i+Root.AoBcomp(AoBcomp_anch_p);
    					while(Bcomp_i<anchor) 
    					{
    						int subanchor = Bcomi+Bcomposition[Bcomp_i];
    						while(Bcomi<subanchor) 
    						{
    							float f = new Random().nextFloat();
    							Root.setMatA(Acomi + inrowsize*Rowi + anchbase*row, counter%10); counter++;//(Acomi + inrowsize*Rowi + anchbase*row)%10; 
    							Bcomi++; 
    							Acomi++;
    						}
    						Bcomp_i++;
    					}
    				}//::::::::::::::::::::::
    				else//======================>>>>  A_comp contains within itself A configuration using B config within Bcomp
    				{//::::::::::::::::::::::
    					int anchor = Bcomp_i+Root.AoBcomp(AoBcomp_anch_p)*-1;
    					while(Bcomp_i<anchor) 
    					{
    						int subanchor = Bcomi+Bcomposition[Bcomp_i];
    						while(Bcomi<subanchor) 
    						{
    							Bcomi++;
    						}
    						Bcomp_i++;
    					}
    					//instead: Bi+= 
    				}//::::::::::::::::::::::
    				//################################
    				AoBcomp_anch_p++;	
    			}
        	}
        	//##################################################################################################################################################################################>
			//===================================================================================================================================================================================>
		    //##################################################################################################################################################################################>
        }
        counter=0;
        for(Insi=0; Insi<Anchor_AoBcomp_size; Insi++) 
        {
        	//matC should be initialized with zeros 
        	for(int Coli = 0 ; Coli < collom; Coli++ )//Ri * collom + Coli
        	{
        		for(int Ri = 0; Ri < row; Ri++ )
        		{
        			float f = new Random().nextFloat();
        			Root.setMatC(Ri + row*Coli + collom*row*Insi, counter%10);// (Ri * collom + Coli)%10;//Ri * collom + Coli;
        		    Root.setC(Ri + row*Coli + collom*row*Insi,counter%10);//(Ri * collom + Coli)%10;    //Ri * collom + Coli;
        			counter++;
        		}
        	}
        }
        counter=0;
        //===========================================================

	}
}
