package neureka.main.exec.gpu.wip;

import java.util.Random;

import com.aparapi.Kernel;

class NAdvancedMatMul extends Kernel 
{
	private float matA[];// = new float[280*2280];
	private float matB[];
	private float matC[];
	private float C[];
	private float B[];
    
    @Constant int row ;
    @Constant int common;
    @Constant int collom;

    int[] Bcomp;// Defines composition of Matrix B (array length is number of components and size of element is size of component)
    int[] AoBcomp;// Defines composition of matrix A over B by defining true/false periods over Bcomp!
    int[] Anchor_AoBcomp;// Defines size of matrix A vector for every input layer
    int[] Anchor_Arowlayer;// Defines 
    
    boolean[] inversed = {true};
  //====================================================
    public int convectionSize() {if(inversed[0]) {return collom*common;}else {return Anchor_AoBcomp.length*collom*row;}}
    
    
    
    public void setA(float[] newA) {matA=newA;}
    public void setB(float[] newB) {matB=newB;}
    public void setC(float[] newC) {matC=newC;}
    
    public void setTestB(float[] newB) {B=newB;}
    public void setTestC(float[] newC) {C=newC;}
    public float[] getTestB() {return B;}
    public float[] getTestC() {return C;}
    
    public float[] getA() {return matA;}
    public float[] getB() {return matB;}
    public float[] getC() {return matC;}

    //public interface calculator{public float[] calculate(float[] input);}
    //calculator addOne;
   
    //==========================================================================
    public NAdvancedMatMul(int row, int collom, int[] Bcomposition, int[] A_Assoc_Composition)
    {
    	
    	//addOne = (float[] input)->{float[] out = {1}; return out;};
    	
    	//Anchor array? :
    	//Is an array of indexes/pointer in order to define another array into e_sub arrays.
    	//This allows for chronoligical access on the subarrays without computational overhead. 
    	
    	//Composition array? :
    	//Defines another array into subarrays as components which are represented as an array of sizes. 
    	//Looping through the e_sub arrays requires computation.
    	
    	Bcomp = Bcomposition;
    	// Defines size of B component for every element:
    	// Sum of components is size of the first B matrix dimension ('height')
    	//-------------------------------------------------------------------------------------------------------
    	
    	AoBcomp = A_Assoc_Composition; 
    	// Stores numbers representing active/inactive connection times for the Bcomp array (positive/negative)
    	// for every input dimension 
    	//==> Distribution is defined by the following anchor array:
    	//-------------------------------------------------------------------------------------------------------


    	//Calculating size of 1 collom layer within matrix B
    	//===================================================
    	int BcompSize=0;
    	for(int i=0; i<Bcomposition.length; i++) {BcompSize += Bcomposition[i];}
    	common = BcompSize;
    	System.out.println("common: "+common+";");
    	//===================================================
    	//Calculating input size from A_toBcomp:
    	//===================================================
    	int comcount = 0;
    	int Insi = 0;
    	System.out.println("Bcomp.length: "+Bcomp.length);
    	for(int i=0; i<AoBcomp.length; i++) 
    	{
    		System.out.println("AoBcomp: ["+i+"]=>["+AoBcomp[i]+"]; comcount: "+comcount);
    		if(comcount==Bcomp.length) 
    		{comcount=0;Insi++;}
    		comcount+=Math.abs(AoBcomp[i]);
    	}
    	int anchoring=0;
    	Anchor_AoBcomp = new int[Insi+1];
    	Insi=0;comcount=0;
    	for(int i=0; i<AoBcomp.length; i++) 
    	{
    		if(comcount==Bcomp.length) 
    		{comcount=0;Anchor_AoBcomp[Insi]=anchoring;Insi++;}
    		comcount+=Math.abs(AoBcomp[i]);
    		anchoring ++;
    	}
    	Anchor_AoBcomp[Insi]=anchoring;
    	//===================================================//The above code works!
    	for(int i=0; i<Anchor_AoBcomp.length; i++) {System.out.println("Anchor_AoBcomp: [->"+Anchor_AoBcomp[i]+"]");}
    	//===================================================

    	//-----------------------------------------
    	Anchor_Arowlayer = new int[Anchor_AoBcomp.length]; // ==> ROW LAYER ANCHOR
    	//Defines anchor points for every input on one row layer.
    	//-----------------------------------------
    	
    	int rowlayer_inputanchoring=0;
    	//=======================================
    	for(Insi=0; Insi<Anchor_AoBcomp.length; Insi++)// ? : ACA -> size == input size !!
    	{
    		int AoBcomp_anch_p;
    		System.out.println("\nNew insi index: "+Insi);
    		System.out.println("=========================");
    		//------------------
    		if(Insi!=0) // Setting anchorpoints:
    		{AoBcomp_anch_p=Anchor_AoBcomp[Insi-1];}else{AoBcomp_anch_p=0;}
    		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    		System.out.println("AoBcomp_anch_p e_set to: "+AoBcomp_anch_p);
    		
    		//-------------------------
    		//Sub array iteration (connection phase discriptor)
    		int Bcomp_i=0;
    		while(AoBcomp_anch_p<(Anchor_AoBcomp[Insi]))// p iterating to the next subarray! 
    		{
    			System.out.println("\n==>> While(AoBcomp_anch_p<(Anchor_AoBcomp[Insi])):("+AoBcomp_anch_p+"<"+(Anchor_AoBcomp[Insi])+")");
    			//################################// This is within one row where Ii & Ri is given!
    			if(0<AoBcomp[AoBcomp_anch_p])// 
    			{//::::::::::::::::::::::
    				int anchor = Bcomp_i+AoBcomp[AoBcomp_anch_p];
    				System.out.println("==>>==>> anchor: "+anchor);
    				while(Bcomp_i<anchor) 
    				{
    					rowlayer_inputanchoring += Bcomp[Bcomp_i];
    					Bcomp_i++;
    				}
    			}//::::::::::::::::::::::
    			else//======================>>>> A_comp contains within itself A configuration using B config within Bcomp
    			{//::::::::::::::::::::::
    				System.out.println("==>>==>> Bcomp_i+=AoBcomp[A_comp_anch]*-1  <->  "+Bcomp_i+"+="+AoBcomp[AoBcomp_anch_p]*-1+"");
    				Bcomp_i+=AoBcomp[AoBcomp_anch_p]*-1;
    			}//::::::::::::::::::::::
    			//################################
    			
    			AoBcomp_anch_p++;	
    		}
    		//-------------------------
    		System.out.println("Anchor_Arowlayer[Insi]=rowlayer_inputanchoring <-> "+Anchor_Arowlayer[Insi]+"="+rowlayer_inputanchoring);
    		Anchor_Arowlayer[Insi]=rowlayer_inputanchoring;//This works!
    	}
    	//=======================================// Yasss!!
    	
    	int A_RowLayerSize = Anchor_Arowlayer[Anchor_Arowlayer.length-1];
    	//=======================================
    	System.out.println("A_RowLayerSize: "+A_RowLayerSize);
    	
    	
        this.row = row;
        this.collom = collom;
        //---------------------------------------------------
        matA = new float [A_RowLayerSize *  row];
        matB = new float [BcompSize * collom ];//* A_comp_anchor.length => Does not have depth: "one to many inputs"
        matC = new float [row * collom * Anchor_AoBcomp.length];
        //---------------------------------------------------
        B = new float[BcompSize * collom];//* A_comp_anchor.length => Does not have depth: "one to many inputs"
        C = new float[row * collom * Anchor_AoBcomp.length];
        //---------------------------------------------------
        
        int counter=0;
        //##################################################################################################################################################################################>
	    //===================================================================================================================================================================================>
	    //##################################################################################################################################################################################>
        for(Insi=0; Insi<Anchor_AoBcomp.length; Insi++) 
        {
        	//Here matrix A is initialized with random number
        	for(int Rowi = 0; Rowi < row; Rowi++ )
        	{
        		int anchbase;
    	    	if(Insi==0) {anchbase=0;}else {anchbase=Anchor_Arowlayer[Insi-1];}
    	    	int inrowsize = Anchor_Arowlayer[Insi]-anchbase;
    			//---------------
    			int AoBcomp_anch_p;// Setting anchor points:
    			if(Insi!=0) {AoBcomp_anch_p=Anchor_AoBcomp[Insi-1];}else {AoBcomp_anch_p=0;}
    			//-------------------------
    			int Bcomp_i=0;
    			int Acomi=0;
    			int Bcomi=0;
    			//Sub array iteration (connection phase discriptor)
    			while(AoBcomp_anch_p<(Anchor_AoBcomp[Insi])) // p iterating to the next subarray!
    			{
    				//################################// This is within one row where Ii & Ri is given!
    				if(0<AoBcomp[AoBcomp_anch_p]) 
    				{//::::::::::::::::::::::
    					int anchor = Bcomp_i+AoBcomp[AoBcomp_anch_p];
    					while(Bcomp_i<anchor) 
    					{
    						int subanchor = Bcomi+Bcomp[Bcomp_i];
    						while(Bcomi<subanchor) 
    						{
    							float f = new Random().nextFloat();
    							matA[Acomi + inrowsize*Rowi + anchbase*row] = counter%10; counter++;//(Acomi + inrowsize*Rowi + anchbase*row)%10; 
    							Bcomi++; 
    							Acomi++;
    						}
    						Bcomp_i++;
    					}
    				}//::::::::::::::::::::::
    				else//======================>>>>  A_comp contains within itself A configuration using B config within Bcomp
    				{//::::::::::::::::::::::
    					int anchor = Bcomp_i+AoBcomp[AoBcomp_anch_p]*-1;
    					while(Bcomp_i<anchor) 
    					{
    						int subanchor = Bcomi+Bcomp[Bcomp_i];
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
        for(Insi=0; Insi<Anchor_AoBcomp.length; Insi++) 
        {
        	//matC should be initialized with zeros 
        	for(int Coli = 0 ; Coli < collom; Coli++ )//Ri * collom + Coli
        	{
        		for(int Ri = 0; Ri < row; Ri++ )
        		{
        			float f = new Random().nextFloat();
        			matC[Ri + row*Coli + collom*row*Insi] = counter%10;// (Ri * collom + Coli)%10;//Ri * collom + Coli;
        		       C[Ri + row*Coli + collom*row*Insi] = counter%10;//(Ri * collom + Coli)%10;    //Ri * collom + Coli;
        			counter++;
        		}
        	}
        }
        counter=0;
        //===========================================================
        // Here matrix B is initialized with random numbers
        for(int Coli = 0 ; Coli < collom; Coli++ )//Comi * collom + Coli
        {
        	for(int Comi = 0; Comi < BcompSize; Comi++ )// B: com X col
        	{
        		float f = new Random().nextFloat();
        		matB[Comi + common * Coli] = counter%10;//(Comi * collom + Coli)%10;//Comi * collom + Coli;//
        		B[Comi + common * Coli] = counter%10;//(Comi * collom + Coli)%10;//Comi * collom + Coli;
        		counter++;
        	}
        }
    }
    
    
    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   //==========================================================================
    public void printResults()
    {
    	for(int i=0; i<Anchor_AoBcomp.length; i++) 
    	{
    		for(int Ri = 0; Ri < row; Ri++ )
    		{
    			for(int C2i = 0 ; C2i < collom; C2i++ )
    			{
    				System.out.print(matC[(Ri * collom * i) + C2i]+"    ");
    			}
    		}
    	}
    }
  //==========================================================================
    public void cpuMatMulCalc()
    {
    	if(inversed[0]) 
    	{
    		System.out.println("\nSequential Execution on CPU (Inversed)");
    		for(int Comi = 0; Comi < common; Comi++)
    			 {
    				 for(int Coli = 0; Coli < collom; Coli++)
    				 {//##################################################################################################################################################################################>
    				  //===================================================================================================================================================================================>
    				  //##################################################################################################################################################################################>
    					 
    						 //=====================================
    						 float value = 0;//Note: Contrary to matrix A and C -> B has no input dimension!	
    				    		for(int Insi=0; Insi<Anchor_AoBcomp.length; Insi++)	
    				    		{	

    				    				int anchbase=0;
    				    				if(Insi==0) {anchbase=0;}else {anchbase=Anchor_Arowlayer[Insi-1];}
    				    				int inrowsize = Anchor_Arowlayer[Insi]-anchbase;
    				    			
    				    				//---------------
    				    				int AoBcomp_anch_p=0;// Setting anchorpoints:
    				    				if(Insi!=0) {AoBcomp_anch_p=Anchor_AoBcomp[Insi-1];}else {AoBcomp_anch_p=0;}
    				    				//-------------------------
    				    		
    				    				int Bcomp_i=0;
    				    				int Acomi=0;
    				    				int Bcomi=0;
    				    				//Sub array iteration (connection phase discriptor)
    				    				while(AoBcomp_anch_p<(Anchor_AoBcomp[Insi])) // p iterating to the next subarray!
    				    				{
    				    					//################################// This is within one row where Ii & Ri is given!
    				    					if(0<AoBcomp[AoBcomp_anch_p]) 
    				    					{//::::::::::::::::::::::
    				    						int anchor = Bcomp_i+AoBcomp[AoBcomp_anch_p];
    				    						while(Bcomp_i<anchor) 
    				    						{
    				    							int subanchor = Bcomi+Bcomp[Bcomp_i];
    				    							while(Bcomi<subanchor) 
    				    							{
    				    								if(Bcomi==Comi)//This is the invers procedure: com*col threading: This already loops through comi! -> checking if Bcmoi==Comi!!...also->some rows of A-transposed might be empty! 
    				    								{
    				    									for(int Rowi=0; Rowi<row; Rowi++)
    				    									{
    				    										//Rowi * common + Comi
    				    										value += matA[Acomi + inrowsize*Rowi + anchbase*row] * matC[Rowi + row*collom*Insi + Coli*row];// Rowi*collom + row*collom*Insi + Coli
    				    										//value += matA[Comi + Rowi*anchbase] * matC[Rowi * collom * Insi + Coli];
    				    									}
    				    								}
    				    								Bcomi++; Acomi++;
    				    							}
    				    							Bcomp_i++;
    				    						}
    				    					}//::::::::::::::::::::::
    				    					else//======================>>>>  A_comp contains within itself A configuration using B config within Bcomp
    				    					{//::::::::::::::::::::::
    				    						int anchor = Bcomp_i+AoBcomp[AoBcomp_anch_p]*-1;
    				    						while(Bcomp_i<anchor) 
    				    						{
    				    							int subanchor = Bcomi+Bcomp[Bcomp_i];
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
    				    		B[Comi + common * Coli] = value;//Comi * collom + Coli
    			 //##################################################################################################################################################################################>
    			 //===================================================================================================================================================================================>
    			 //##################################################################################################################################################################################>
    			 }
    		 }
    	}
    	else
    	{
    		//'#############################################################################################################################################
    		//THIS CODE SECTION WORKS BABY! -> Calculations are correct!!!!!
    		//'#############################################################################################################################################
    		System.out.println("\nSequential Execution on CPU\n===============================");
    		for(int Insi=0; Insi<Anchor_AoBcomp.length; Insi++) 
    		{
    			
    			for(int Rowi = 0;Rowi < row; Rowi++)
    			{
    				System.out.println("\nRowi: "+Rowi);
    				for(int Coli = 0; Coli < collom; Coli++)
    				{//##################################################################################################################################################################################>
    				 //===================================================================================================================================================================================>
    				 //##################################################################################################################################################################################>
    					
    					
    					System.out.println("Coli: "+Coli);
    					System.out.println("###################################");

			    			int anchbase=0;
			    	    	if(Insi==0) {anchbase=0;}else {anchbase=Anchor_Arowlayer[Insi-1];}
			    	    	int inrowsize = Anchor_Arowlayer[Insi]-anchbase;
			    	    	
			    			float value = 0;
			    			//---------------
			    			int AoBcomp_anch_p;// Setting anchorpoints:
			    			if(Insi!=0) {AoBcomp_anch_p=Anchor_AoBcomp[Insi-1];}else {AoBcomp_anch_p=0;}
			    			//-------------------------
			    		
			    			int Bcomp_i=0;
			    			int Acomi=0;//????????
			    			int Bcomi=0;
			    			//Sub array iteration (connection phase discriptor)
			    			while(AoBcomp_anch_p<(Anchor_AoBcomp[Insi])) // p iterating to the next subarray!
			    			{//System.out.println("\n==>> While(AoBcomp_anch_p<(Anchor_AoBcomp[Insi])):("+AoBcomp_anch_p+"<"+(Anchor_AoBcomp[Insi])+")");
			    				//################################// This is within one row where Ii & Ri is given!
			    				
			    				//System.out.println("==>> if(0<AoBcomp[AoBcomp_anch_p]):(0<"+AoBcomp[AoBcomp_anch_p]+")");
			    				if(0<AoBcomp[AoBcomp_anch_p]) 
			    				{//::::::::::::::::::::::
			    					int anchor = Bcomp_i+AoBcomp[AoBcomp_anch_p];
			    					while(Bcomp_i<anchor) 
			    					{//System.out.println("\n==>>==>> While(Bcomp_i<(anchor)):("+Bcomp_i+"<"+anchor+")");
			    						int subanchor = Bcomi+Bcomp[Bcomp_i];
			    						while(Bcomi<subanchor) 
			    						{//System.out.println("\n==>>==>>==>> While(Bcomi<(subanchor)):("+Bcomi+"<"+subanchor+")");
			    							//Rowi * common + Comi
			    							//System.out.print("Acomi: "+Acomi+"; Bcomi: "+Bcomi+"; anchbase: "+anchbase+"; Rowi: "+Rowi);
			    							//System.out.println("; collom: "+collom+"; Insii: "+Insi+"; Coli: "+Coli+"; inrowsize: "+inrowsize);
			    							//	System.out.println("value:("+value+")+=A:("+matA[Acomi + inrowsize*Rowi + anchbase*row]+")*B:("+matB[Bcomi + common*Coli]+")");
			    							value += matA[Acomi + inrowsize*Rowi + anchbase*row] * matB[Bcomi + common*Coli]; //Bcomi * collom + Coli
			    							Bcomi++; 
			    							Acomi++;
			    						}
			    						Bcomp_i++;
			    					}
			    				}//::::::::::::::::::::::
			    				else//======================>>>>  A_comp contains within itself A configuration using B config within Bcomp
			    				{//::::::::::::::::::::::
			    					int anchor = Bcomp_i+AoBcomp[AoBcomp_anch_p]*-1;
			    					while(Bcomp_i<anchor) 
			    					{
			    						int subanchor = Bcomi+Bcomp[Bcomp_i];
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
			    			C[Insi*row*collom + Rowi + row*Coli] = value;//Insi*row*collom + Rowi*collom + Coli
			    			
			    	//##################################################################################################################################################################################>
			    	//===================================================================================================================================================================================>
			    	//##################################################################################################################################################################################>
    				}
    			}
    		}
    		//'#############################################################################################################################################
    		//THIS CODE SECTION WORKS BABY!
    		//'#############################################################################################################################################
    	}
    }
    @Override
    public void run() 
    {
    	//====================================================
    	//Inversed: Thread for every col*com layer element! ->  inp->inside: ->looping: row->inside: addressing composition ->checking if current com is used 
    	//Normal: Thread for every row*col*inp layer element! -> looping through:  com! -> composition!
    	//====================================================
    	if(inversed[0]) //At x C = B  // => Range must be: row*col*inp
    	{//------------------------------------------------------
    		//Defining index space:
    	    //int Insi = getGlobalId()%Anchor_AoBcomp.length;
    	    int Coli = (getGlobalId()) % collom; 
    		int Comi = (getGlobalId()-Coli) / collom;
    		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    		float value = 0;//Note: Contrary to matrix A and C -> B has no input dimension!	
    		for(int Insi=0; Insi<Anchor_AoBcomp.length; Insi++)	
    		{	
    				int anchbase=0;
    				if(Insi==0) {anchbase=0;}else {anchbase=Anchor_Arowlayer[Insi-1];}
    				int inrowsize = Anchor_Arowlayer[Insi]-anchbase;
    				//-----------------------------------------------------------------------------
    				int AoBcomp_anch_p=0;// Setting anchorpoints:
    				if(Insi!=0) {AoBcomp_anch_p=Anchor_AoBcomp[Insi-1];}else {AoBcomp_anch_p=0;}
    				//-----------------------------------------------------------------------------
    				int Bcomp_i=0;
    				int Acomi=0;
    				int Bcomi=0;
    				//Sub array iteration (connection phase discriptor)
    				while(AoBcomp_anch_p<(Anchor_AoBcomp[Insi])) // p iterating to the next subarray!
    				{
    					//################################// This is within one row where Ii & Ri is given!
    					if(0<AoBcomp[AoBcomp_anch_p]) 
    					{//::::::::::::::::::::::
    						int anchor = Bcomp_i+AoBcomp[AoBcomp_anch_p];
    						while(Bcomp_i<anchor) 
    						{
    							int subanchor = Bcomi+Bcomp[Bcomp_i];
    							while(Bcomi<subanchor) 
    							{
    								if(Bcomi==Comi)//This is the invers procedure: com*col threading: This already loops through comi! -> checking if Bcmoi==Comi!!...also->some rows of A-transposed might be empty! 
    								{
    									for(int Rowi=0; Rowi<row; Rowi++)
    									{
    										value += matA[Acomi + inrowsize*Rowi + anchbase*row] * matC[Rowi + row*collom*Insi + Coli*row];// Rowi*collom + row*collom*Insi + Coli
    									}
    								}
    								Bcomi++; Acomi++;
    							}
    							Bcomp_i++;
    						}
    					}//::::::::::::::::::::::
    					else//======================>>>>  A_comp contains within itself A configuration using B config within Bcomp
    					{//::::::::::::::::::::::
    						int anchor = Bcomp_i+AoBcomp[AoBcomp_anch_p]*-1;
    						while(Bcomp_i<anchor) 
    						{
    							int subanchor = Bcomi+Bcomp[Bcomp_i];
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
    		//matC[Rowi * collom + Coli] = value;
    		matB[Comi + Coli*common] = value;//Comi * collom + Coli
    	    //################################
    	}//------------------------------------------------------
    	else // A x B = C  // => Range must be: row*com*inp
    	{//------------------------------------------------------
    		int lvl = getGlobalId()%(row*collom);
    		int Insi = (getGlobalId()-lvl)/(row*collom);//getGlobalId()/Anchor_AoBcomp.length;
        	int Coli = (lvl) % collom; 
    		int Rowi = (lvl-Coli) /collom;
    		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    		int anchbase=0;
    			
    	    if(Insi==0) {anchbase=0;}else {anchbase=Anchor_Arowlayer[Insi-1];}
    		int inrowsize = Anchor_Arowlayer[Insi]-anchbase;
    	    	
    		float value = 0;
    		//---------------
    		int AoBcomp_anch_p=0;// Setting anchorpoints:
    		if(Insi!=0) {AoBcomp_anch_p=Anchor_AoBcomp[Insi-1];}else {AoBcomp_anch_p=0;}
    		//-------------------------
    		int Bcomp_i=0;
    		int Acomi=0;
    		int Bcomi=0;
    		//Sub array iteration (connection phase discriptor)
    		while(AoBcomp_anch_p<(Anchor_AoBcomp[Insi])) // p iterating to the next subarray!
    		{
    			//################################// This is within one row where Ii & Ri is given!
    			if(0<AoBcomp[AoBcomp_anch_p]) 
    			{//::::::::::::::::::::::
    				int anchor = Bcomp_i+AoBcomp[AoBcomp_anch_p];
    				while(Bcomp_i<anchor) 
    				{
    					int subanchor = Bcomi+Bcomp[Bcomp_i];
    					while(Bcomi<subanchor) 
    					{
    						value += matA[Acomi + inrowsize*Rowi + anchbase*row] * matB[Bcomi + common*Coli]; //Bcomi * collom + Coli
    						Bcomi++; Acomi++;
    					}
    					Bcomp_i++;
    				}
    			}//::::::::::::::::::::::
    			else//======================>>>>  A_comp contains within itself A configuration using B config within Bcomp
    			{//::::::::::::::::::::::
    				int anchor = Bcomp_i+AoBcomp[AoBcomp_anch_p]*-1;
    				while(Bcomp_i<anchor) 
    				{
    					int subanchor = Bcomi+Bcomp[Bcomp_i];
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
    		matC[Insi*row*collom + Rowi + row*Coli] = value;//Insi*row*collom + Rowi*collom + Coli
    	}//------------------------------------------------------
    }
    //MATRIX PRINTING:
    //============================================
    public void printAll() {printA();printC();printAT();printB();}
    //------------------
     public void printAT()
    {
    	 System.out.println("At["+Anchor_AoBcomp.length+" X "+row+" X "+Anchor_Arowlayer[Anchor_Arowlayer.length-1]+"]:\n================================");
    	 for(int Insi=0; Insi<Anchor_AoBcomp.length; Insi++) 
    	 {
    		 System.out.println("At[("+Insi+") X "+row+" X "+Anchor_Arowlayer[Anchor_Arowlayer.length-1]+"]:\n-----------------");
    		 for(int Comi = 0; Comi < common ; Comi++)
    		 {
    			System.out.print("->[Comi]=>["+Comi+"]:  ");
		    	int anchbase;
		    	if(Insi==0) {anchbase=0;}else {anchbase=Anchor_Arowlayer[Insi-1];}
		    	int inrowsize = Anchor_Arowlayer[Insi]-anchbase;
		    	//---------------
		    	int AoBcomp_anch_p;// Setting anchorpoints:
		    	if(Insi!=0) {AoBcomp_anch_p=Anchor_AoBcomp[Insi-1];}else {AoBcomp_anch_p=0;}
		    	//-------------------------
		    	int Bcomp_i=0;
		    	int Acomi=0;
		    	int Bcomi=0;
		    	//Sub array iteration (connection phase discriptor)
		    	while(AoBcomp_anch_p<(Anchor_AoBcomp[Insi])) // p iterating to the next subarray!
		    	{
		  			//################################// This is within one row where Ii & Ri is given!
		    		if(0<AoBcomp[AoBcomp_anch_p]) 
		    		{//::::::::::::::::::::::
		    			int anchor = Bcomp_i+AoBcomp[AoBcomp_anch_p];
		    			while(Bcomp_i<anchor) 
		    			{ 
		    				int subanchor = Bcomi+Bcomp[Bcomp_i];
		    				while(Bcomi<subanchor) 
		    				{
		    					if(Bcomi==Comi)//This is the invers procedure: com*col threading: This already loops through comi! -> checking if Bcmoi==Comi!!...also->some rows of A-transposed might be empty! 
		    					{
		    						for(int Rowi=0; Rowi<row; Rowi++)
		    						{
		    							System.out.print("["+matA[Acomi + inrowsize*Rowi + anchbase*row]+"]");//...row*Insi
		    						}
		    					}
		    					Bcomi++; Acomi++;
		    				}
		    				Bcomp_i++;
		    			}
		    		}//::::::::::::::::::::::
		    		else//======================>>>>  A_comp contains within itself A configuration using B config within Bcomp
		    		{//::::::::::::::::::::::
		    			int anchor = Bcomp_i+AoBcomp[AoBcomp_anch_p]*-1;
		    			while(Bcomp_i<anchor) 
		    			{
		    				int subanchor = Bcomi+Bcomp[Bcomp_i];
		    				while(Bcomi<subanchor) 
		    				{
		    					if(Bcomi==Comi)//This is the invers procedure: com*col threading: This already loops through comi! -> checking if Bcmoi==Comi!!...also->some rows of A-transposed might be empty! 
		    					{
		    						for(int Rowi=0; Rowi<row; Rowi++)
	    							{
	    								System.out.print("[###]");//...row*Insi
	    							}
		    					}
		    					Bcomi++;
		    				}
		    				Bcomp_i++;
		    			}
		    			//instead: Bi+= 
		    		}//::::::::::::::::::::::
		    		//################################
		    		AoBcomp_anch_p++;			
		    	}	
		    	System.out.println("");		
    		 }
    		 System.out.println('\n');
    	 }
    }
   //------------------
    public void printA()
    {
    	System.out.println("A["+Anchor_Arowlayer.length+" X "+row+" X "+Anchor_Arowlayer[Anchor_Arowlayer.length-1]+"]:\n================================");
    	for(int Insi=0; Insi<Anchor_AoBcomp.length; Insi++) 
    	{
    		System.out.println("A[("+Insi+") X "+row+" X "+Anchor_Arowlayer[Anchor_Arowlayer.length-1]+"]:\n-----------------");

    		for(int Rowi = 0; Rowi < row ; Rowi++)
    		{
    			System.out.print("->[Rowi]=>["+Rowi+"]:  ");
    			
    			int anchbase;
    	    	if(Insi==0) {anchbase=0;}else {anchbase=Anchor_Arowlayer[Insi-1];}
    	    	int inrowsize = Anchor_Arowlayer[Insi]-anchbase;
    	    	
    			//---------------
    			int AoBcomp_anch_p;// Setting anchor points:
    			if(Insi!=0) {AoBcomp_anch_p=Anchor_AoBcomp[Insi-1];}else{AoBcomp_anch_p=0;}
    			//-------------------------
    			int Acomi=0;
    			int Bcomi=0;
    			int Bcomp_i=0;
    			//Sub array iteration (connection phase discriptor)
    			while(AoBcomp_anch_p<(Anchor_AoBcomp[Insi])) // p iterating to the next e_sub-array!
    			{
    				//################################// This is within one row where Ii & Ri is given!
    				
    				if(0<AoBcomp[AoBcomp_anch_p]) 
    				{//::::::::::::::::::::::
    					int anchor = Bcomp_i+AoBcomp[AoBcomp_anch_p];
    					while(Bcomp_i<anchor) 
    					{
    						int subanchor = Bcomi+Bcomp[Bcomp_i];
    						while(Bcomi<subanchor) 
    						{
    			    			System.out.print("["+matA[Acomi + Rowi*inrowsize+anchbase*row]+"]");
    							Bcomi++; Acomi++;
    						}
    						Bcomp_i++;
    					}
    				}//::::::::::::::::::::::
    				else//======================>>>>  A_comp contains within itself A configuration using B config within Bcomp
    				{//::::::::::::::::::::::
    					int anchor = Bcomp_i+AoBcomp[AoBcomp_anch_p]*-1;
    					while(Bcomp_i<anchor) 
    					{
    						int subanchor = Bcomi+Bcomp[Bcomp_i];
    						while(Bcomi<subanchor) 
    						{
    			    			System.out.print("[###]");
    							Bcomi++;
    						}
    						Bcomp_i++;
    					}
    				}//::::::::::::::::::::::
    				//################################
    				AoBcomp_anch_p++;	
    			}
    			System.out.println("");
    		}
    		System.out.println('\n');
    	}
    }
    public void printB()
    {
    	System.out.println("B["+common+" X "+collom+"]:\n================================");
    	
    		for(int Bi = 0; Bi < common * collom ; Bi++)
    		{
    			if(Bi%collom==0) {System.out.print("\n->[Comi]=>["+Bi/collom+"]:  ");}
    			System.out.print("["+B[((int)(((Bi)/(collom)))) + common*((int)((Bi%collom)))]+"]");
    		}
    		System.out.println('\n');
    }
    public void printC() 
    {
    	System.out.println("C["+Anchor_AoBcomp.length+" X "+row+" X "+collom+"]:\n================================");
    	for(int Insi=0; Insi<Anchor_AoBcomp.length; Insi++) 
    	{
    		System.out.print("C[("+Insi+") X "+row+" X "+collom+"]:\n-----------------");
    		for(int Ci = 0; Ci < row * collom ; Ci++)
    		{
    			if(Ci%collom==0) {System.out.print("\n->[Rowi]=>["+Ci/collom+"]:  ");}///collom
    			System.out.print("["+C[Insi*row*collom + ((int)(((Ci)/(collom)))) + row*((int)((Ci%collom)))]+"]");     
    		}
    		System.out.println('\n');
    	}
    }
  //==========================================================================
    
    //Comparing CPU C-Matrix with with GPU C-Matrix!
    public void compareResults()
    {
    	if(inversed[0])
    	{
    			boolean equal = true;
    			for(int Ci = 0; Ci < common * collom ; Ci++)
    			{
    				if(Math.abs(matB[Ci]-B[Ci])>0.001)
    				{
    					System.out.println("B:"+B[Ci]+"; matB:"+matB[Ci]+";");
    					equal = false; break;
    				}
    			}
    			if(equal==false) 
    			{System.out.println("Results are not equal");}
    			else	   
    			{System.out.println("Results are equal.. Tested thoroughly!!!");}
    	}
    	else
    	{
    			boolean equal = true;
    			for(int Ci = 0; Ci < row * collom * Anchor_AoBcomp.length; Ci++)
    			{
    				
    				if(Math.abs(matC[Ci]-C[Ci])>0.001)
    				{
    					System.out.println("C:"+C[Ci]+"; matC:"+matC[Ci]+";");
    					equal = false; break;
    				}
    			}
    			if(equal==false) 
    			{System.out.println("Results are not equal");}
    			else	   
    			{System.out.println("Results are equal.. Tested thoroughly!!!");}
    		}
    }
  //---------------------------------------------------
}
