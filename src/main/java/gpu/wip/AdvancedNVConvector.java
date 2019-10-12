package gpu.wip;

import java.util.Random;

import com.aparapi.Kernel;
import com.aparapi.Kernel.Constant;

import com.aparapi.ProfileInfo;
import com.aparapi.Range;

public class AdvancedNVConvector extends Kernel
{
	
	//Secret fields!
	private int[] AoBcomp;// Defines composition of matrix A over B by defining true/false periods over Bcomp!
	private int[] Anchor_AoBcomp;// Defines size of matrix A vector for every input layer
	private int[] Anchor_Arowlayer;// Defines row layer anchors
	private float matA[];
	private float matC[];
	private float C[];
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    public float[] matA()      {return matA;}
    public float   matA(int i) {return matA[i];}
    public void    setMatA(float[] newA) {matA=newA;}
    public void    setMatA(int i, float value) {matA[i]=value;}
  //-----------------------------------------------
    public float[] matC()      {return matC;}
    public float   matC(int i) {return matC[i];}
    public void    setMatC(float[] newC) {matC=newC;}
    public void    setMatC(int i, float value) {matC[i]=value;}
  //-----------------------------------------------
    public float[] C()      {return C;}
    public float   C(int i) {return C[i];}
    public void	   setC(float[] newC) {C=newC;}
    public void    setC(int i, float value) {C[i]=value;} 
  //-----------------------------------------------
    public int[] AoBcomp()      {return AoBcomp;}
    public int   AoBcomp(int i) {return AoBcomp[i];}
    public void  addAoBcomp(int index, int[] newAoB) {AoBcomp=newAoB;}
    public void setAoBcomp(int i, int value) {AoBcomp[i]=value;}
    
    public int[] Anchor_AoBcomp()      {return Anchor_AoBcomp;}
    public int   Anchor_AoBcomp(int i) {return Anchor_AoBcomp[i];}
    public void  addAnchor_AoBcomp(int index, int[] newAoBanchor) {Anchor_AoBcomp=newAoBanchor;}
    public void  setAnchor_AoBcomp(int i, int value) {Anchor_AoBcomp[i]=value;}
    
    public int[] Anchor_Arowlayer()      {return Anchor_Arowlayer;}
    public int   Anchor_Arowlayer(int i) {return Anchor_Arowlayer[i];}
    public void  addAnchor_Arowlayer(int index, int[] newAnchor) {Anchor_Arowlayer = newAnchor;}
    public void  setAnchor_Arowlayer(int i, int value) {Anchor_Arowlayer[i]=value;}
  //------------------------------------------------------------------
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	//---------------------------------
	//-----------------------
	//--------------
	
	private NVCElement[] E = new NVCElement[1];
	private float matB[];
	private float B[];
    
    @Constant int common;
    @Constant int collom;
    int[] Bcomp;// Defines composition of Matrix B (array length is number of components and size of element is size of component)

    boolean[] inversed = {true};
  //====================================================
    public int els() {return E.length;}
    public int convectionSize(int i) 
    {
    	if(inversed[0]) 
    	{return collom*common;}
    	else 
    	{return E[i].ins()*collom*E[i].row();}
    }
    public void setA(int ei, float[] newA) {E[ei].setMatA(newA);}
    public void setC(int ei, float[] newC) {E[ei].setMatC(newC);}
    public void setB(float[] newB) {matB=newB;}
    
    public void setTestB(float[] newB) {B=newB;}
    public void setTestC(int ei, float[] newC) {E[ei].setC(newC);}
    public float[] getTestB() {return B;}
    public float[] getTestC(int ei) {return E[ei].C();}
    
    public float[] getB() {return matB;}
    public float[] getA(int ei) {return E[ei].matA();}
    public float[] getC(int ei) {return E[ei].matC();}

    //public interface calculator{public float[] calculate(float[] input);}
    //calculator addOne;
    //==========================================================================
    public AdvancedNVConvector(int row, int collom, int[] Bcomposition, int[][] AoverBComposition)
    {
    	E[0] = new NVCElement(this, row, collom, AoverBComposition[0], Bcomposition);
    	//addOne = (float[] input)->{float[] out = {1}; return out;};
    	
    	//Anchor array? :
    	//Is an array of indexes/pointer in order to define another array into sub arrays.
    	//This allows for chronological access on the sub-arrays without computational overhead. 
    	
    	//Composition array? :
    	//Defines another array into subarrays as components which are represented as an array of sizes. 
    	//Looping through the sub arrays requires computation.
    	
    	Bcomp = Bcomposition;
    	// Defines size of B component for every element:
    	// Sum of components is size of the first B matrix dimension ('height')
    	//-------------------------------------------------------------------------------------------------------
    	
    	//AoBcomp = AoverBComposition; 
    	// Stores numbers representing active/inactive connection times for the Bcomp array (positive/negative)
    	// for every input dimension 
    	//==> Distribution is defined by the following anchor array:
    	//-------------------------------------------------------------------------------------------------------

    	//Calculating size of 1 collom layer within matrix B
    	//===================================================
    	int BcompSize = 0;
    	for(int i=0; i<Bcomposition.length; i++) 
    	{
    		BcompSize += Bcomposition[i];
    	}
    	common = BcompSize;
    	System.out.println("common: "+common+";");
    	//===================================================
    	
        this.collom = collom;
        //---------------------------------------------------
        matB = new float [BcompSize * collom ];//* A_comp_anchor.length => Does not have depth: "one to many inputs"
        //---------------------------------------------------
        B = new float[BcompSize * collom];//* A_comp_anchor.length => Does not have depth: "one to many inputs"
        //---------------------------------------------------
        
        int counter=0;
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
    public void printResults(int ei)
    {
    	for(int i=0; i<E[ei].ins(); i++) 
    	{
    		for(int Ri = 0; Ri < E[ei].row(); Ri++ )
    		{
    			for(int C2i = 0 ; C2i < collom; C2i++ )
    			{
    				System.out.print(E[ei].matC((Ri * collom * i) + C2i)+"    ");
    			}
    		}
    	}
    }
  //==========================================================================
    public void cpuMatMulCalc()
    {
    	if(inversed[0]) 
    	{
    		for(int ei=0; ei<els(); ei++)
	    	{
	    		System.out.println("\nSequential Execution on CPU (Inversed)");
	    		for(int Comi = 0; Comi < common; Comi++)
	    			 {
	    				 for(int Coli = 0; Coli < collom; Coli++)
	    				 {
	    					 //##################################################################################################################################################################################>
	    					 //===================================================================================================================================================================================>
	    					 //##################################################################################################################################################################################>
	    					 
	    					 //=====================================
	    					 float value = 0;//Note: Contrary to matrix A and C -> B has no input dimension!	
	    				     for(int Insi=0; Insi<E[ei].ins(); Insi++)	
	    				     {	
	    				    	 int anchbase=0;
	    				    	 if(Insi==0) 
	    				    	{anchbase=0;}
	    				    	else 
	    				    	{anchbase=E[ei].Anchor_Arowlayer(Insi-1);}
	    				    	int inrowsize = E[ei].Anchor_Arowlayer(Insi)-anchbase;
	    				 
	    				    	//---------------
	    				    	int AoBcomp_anch_p=0;// Setting anchorpoints:
	    				    	if(Insi!=0) {AoBcomp_anch_p=E[ei].Anchor_AoBcomp(Insi-1);}else {AoBcomp_anch_p=0;}
	    				    	//-------------------------
	    				    	int Bcomp_i=0;
	    				    	int Acomi=0;
	    				    	int Bcomi=0;
	    				    	//Sub array iteration (connection phase discriptor)
	    				    	while(AoBcomp_anch_p<(E[ei].Anchor_AoBcomp(Insi))) // p iterating to the next subarray!
	    				    	{
	    				    		//################################// This is within one row where Ii & Ri is given!
	    				    		if(0<E[ei].AoBcomp(AoBcomp_anch_p)) 
	    				    		{//::::::::::::::::::::::
	    				    			int anchor = Bcomp_i+E[ei].AoBcomp(AoBcomp_anch_p);
	    				    			while(Bcomp_i<anchor) 
	    				    			{
	    				    				int subanchor = Bcomi+Bcomp[Bcomp_i];
	    				    				while(Bcomi<subanchor) 
	    				    				{
	    				    					if(Bcomi==Comi)//This is the invers procedure: com*col threading: This already loops through comi! -> checking if Bcmoi==Comi!!...also->some rows of A-transposed might be empty! 
	    				    					{
	    				    						for(int Rowi=0; Rowi<E[ei].row(); Rowi++)
	    				    						{
	    				    							//Rowi * common + Comi
	    				    							value += E[ei].matA(Acomi + inrowsize*Rowi + anchbase*E[ei].row()) * E[ei].matC(Rowi + E[ei].row()*collom*Insi + Coli*E[ei].row());// Rowi*collom + row*collom*Insi + Coli
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
	    				    			int anchor = Bcomp_i+E[ei].AoBcomp(AoBcomp_anch_p)*-1;
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
    	}
    	else
    	{
    		//'#############################################################################################################################################
    		//THIS CODE SECTION WORKS BABY! -> Calculations are correct!!!!!
    		//'#############################################################################################################################################
	    	for(int ei=0; ei<els(); ei++)
	    	{	
    			System.out.println("\nSequential Execution on CPU\n===============================");
	    		for(int Insi=0; Insi<E[ei].ins(); Insi++) 
	    		{
	    			for(int Rowi = 0;Rowi < E[ei].row(); Rowi++)
	    			{
	    				System.out.println("\nRowi: "+Rowi);
	    				for(int Coli = 0; Coli < collom; Coli++)
	    				{//##################################################################################################################################################################################>
	    				 //===================================================================================================================================================================================>
	    				 //##################################################################################################################################################################################>
	    					System.out.println("Coli: "+Coli);
	    					System.out.println("###################################");
	
				    		int anchbase=0;
				    	    if(Insi==0) {anchbase=0;}else {anchbase=E[ei].Anchor_Arowlayer(Insi-1);}
				    	    int inrowsize = E[ei].Anchor_Arowlayer(Insi)-anchbase;
				    	    	
				    		float value = 0;
				    		//---------------
				    		int AoBcomp_anch_p;// Setting anchorpoints:
				    		if(Insi!=0) {AoBcomp_anch_p=E[ei].Anchor_AoBcomp(Insi-1);}else {AoBcomp_anch_p=0;}
				    		//-------------------------
				    		
				    		int Bcomp_i=0;
				    		int Acomi=0;//????????
				    		int Bcomi=0;
				    			//Sub array iteration (connection phase discriptor)
				    		while(AoBcomp_anch_p<(E[ei].Anchor_AoBcomp(Insi))) // p iterating to the next subarray!
				    		{//System.out.println("\n==>> While(AoBcomp_anch_p<(Anchor_AoBcomp()[Insi])):("+AoBcomp_anch_p+"<"+(Anchor_AoBcomp()[Insi])+")");
				    			//################################// This is within one row where Ii & Ri is given!
				    				
				    			//System.out.println("==>> if(0<AoBcomp[AoBcomp_anch_p]):(0<"+AoBcomp[AoBcomp_anch_p]+")");
				    			if(0<E[ei].AoBcomp(AoBcomp_anch_p)) 
				    			{//::::::::::::::::::::::
				    				int anchor = Bcomp_i+E[ei].AoBcomp(AoBcomp_anch_p);
				    				while(Bcomp_i<anchor) 
				    				{//System.out.println("\n==>>==>> While(Bcomp_i<(anchor)):("+Bcomp_i+"<"+anchor+")");
				    					int subanchor = Bcomi+Bcomp[Bcomp_i];
				    					while(Bcomi<subanchor) 
				    					{
				    						//System.out.println("\n==>>==>>==>> While(Bcomi<(subanchor)):("+Bcomi+"<"+subanchor+")");
				    						//Rowi * common + Comi
				    						//System.out.print("Acomi: "+Acomi+"; Bcomi: "+Bcomi+"; anchbase: "+anchbase+"; Rowi: "+Rowi);
				    						//System.out.println("; collom: "+collom+"; Insii: "+Insi+"; Coli: "+Coli+"; inrowsize: "+inrowsize);
				    						//	System.out.println("value:("+value+")+=A:("+matA[Acomi + inrowsize*Rowi + anchbase*row]+")*B:("+matB[Bcomi + common*Coli]+")");
				    						value += E[ei].matA(Acomi + inrowsize*Rowi + anchbase*E[ei].row()) * matB[Bcomi + common*Coli]; //Bcomi * collom + Coli
				    						Bcomi++; 
				    						Acomi++;
				    					}
				    					Bcomp_i++;
				    				}
				    			}//::::::::::::::::::::::
				    			else//======================>>>>  A_comp contains within itself A configuration using B config within Bcomp
				    			{//::::::::::::::::::::::
				    				int anchor = Bcomp_i+E[ei].AoBcomp(AoBcomp_anch_p)*-1;
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
				    		E[ei].setC(Insi*E[ei].row()*collom + Rowi + E[ei].row()*Coli, value);//Insi*row*collom + Rowi*collom + Coli		
				    	//##################################################################################################################################################################################>
				    	//===================================================================================================================================================================================>
				    	//##################################################################################################################################################################################>
	    				}
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
	    	for(int ei=0; ei<els(); ei++)
	    	{
	    		//Defining index space:
	    	    //int Insi = getGlobalId()%Anchor_AoBcomp().length;
	    	    int Coli = (getGlobalId()) % collom; 
	    		int Comi = (getGlobalId()-Coli) / collom;
	    		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	    		float value = 0;//Note: Contrary to matrix A and C -> B has no input dimension!	
	    		for(int Insi=0; Insi<E[ei].ins(); Insi++)	
	    		{	
	    			int anchbase=0;
	    			if(Insi==0) 
	    			{anchbase=0;}
	    			else 
	    			{anchbase=E[ei].Anchor_Arowlayer(Insi-1);}
	    			int inrowsize = E[ei].Anchor_Arowlayer(Insi)-anchbase;
	    			//-----------------------------------------------------------------------------
	    			int AoBcomp_anch_p=0;// Setting anchorpoints:
	    			if(Insi!=0) {AoBcomp_anch_p=E[ei].Anchor_AoBcomp(Insi-1);}else {AoBcomp_anch_p=0;}
	    			//-----------------------------------------------------------------------------
	    			int Bcomp_i=0;
	    			int Acomi=0;
	    			int Bcomi=0;
	    			//Sub array iteration (connection phase discriptor)
	    			while(AoBcomp_anch_p<(E[ei].Anchor_AoBcomp(Insi))) // p iterating to the next subarray!
	    			{
	    				//################################// This is within one row where Ii & Ri is given!
	    				if(0<E[ei].AoBcomp(AoBcomp_anch_p)) 
	    				{//::::::::::::::::::::::
	    					int anchor = Bcomp_i+E[ei].AoBcomp(AoBcomp_anch_p);
	    					while(Bcomp_i<anchor) 
	    					{
	    						int subanchor = Bcomi+Bcomp[Bcomp_i];
	    						while(Bcomi<subanchor) 
	    						{
	    							if(Bcomi==Comi)//This is the invers procedure: com*col threading: This already loops through comi! -> checking if Bcmoi==Comi!!...also->some rows of A-transposed might be empty! 
	    							{
	    								for(int Rowi=0; Rowi<E[ei].row(); Rowi++)
	    								{
	    									value += E[ei].matA(Acomi + inrowsize*Rowi + anchbase*E[ei].row()) * E[ei].matC(Rowi + E[ei].row()*collom*Insi + Coli*E[ei].row());// Rowi*collom + row*collom*Insi + Coli
	    								}
	    							}
	    							Bcomi++; Acomi++;
	    						}
	    						Bcomp_i++;
	    					}
	    				}//::::::::::::::::::::::
	    				else//======================>>>>  A_comp contains within itself A configuration using B config within Bcomp
	    				{//::::::::::::::::::::::
	    					int anchor = Bcomp_i+E[ei].AoBcomp(AoBcomp_anch_p)*-1;
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
	        }
	    }//------------------------------------------------------
	    else // A x B = C  // => Range must be: row*com*inp
	    {//------------------------------------------------------
	    	for(int ei=0; ei<els(); ei++)
		    {
	    		int lvl = getGlobalId()%(E[ei].row()*collom);
	    		int Insi = (getGlobalId()-lvl)/(E[ei].row()*collom);//getGlobalId()/Anchor_AoBcomp().length;
	        	int Coli = (lvl) % collom; 
	    		int Rowi = (lvl-Coli) /collom;
	    		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	    		int anchbase=0;
	    			
	    	    if(Insi==0) {anchbase=0;}else {anchbase=E[ei].Anchor_Arowlayer(Insi-1);}
	    		int inrowsize = E[ei].Anchor_Arowlayer(Insi)-anchbase;
	    	    	
	    		float value = 0;
	    		//---------------
	    		int AoBcomp_anch_p=0;// Setting anchorpoints:
	    		if(Insi!=0) {AoBcomp_anch_p=E[ei].Anchor_AoBcomp(Insi-1);}else {AoBcomp_anch_p=0;}
	    		//-------------------------
	    		int Bcomp_i=0;
	    		int Acomi=0;
	    		int Bcomi=0;
	    		//Sub array iteration (connection phase discriptor)
	    		while(AoBcomp_anch_p<(E[ei].Anchor_AoBcomp(Insi))) // p iterating to the next subarray!
	    		{
	    			//################################// This is within one row where Ii & Ri is given!
	    			if(0<E[ei].AoBcomp(AoBcomp_anch_p)) 
	    			{//::::::::::::::::::::::
	    				int anchor = Bcomp_i+E[ei].AoBcomp(AoBcomp_anch_p);
	    				while(Bcomp_i<anchor) 
	    				{
	    					int subanchor = Bcomi+Bcomp[Bcomp_i];
	    					while(Bcomi<subanchor) 
	    					{
	    						value += E[ei].matA(Acomi + inrowsize*Rowi + anchbase*E[ei].row()) * matB[Bcomi + common*Coli]; //Bcomi * collom + Coli
	    						Bcomi++; Acomi++;
	    					}
	    					Bcomp_i++;
	    				}
	    			}//::::::::::::::::::::::
	    			else//======================>>>>  A_comp contains within itself A configuration using B config within Bcomp
	    			{//::::::::::::::::::::::
	    				int anchor = Bcomp_i+E[ei].AoBcomp(AoBcomp_anch_p)*-1;
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
	    		E[ei].setMatC(Insi*E[ei].row()*collom + Rowi + E[ei].row()*Coli, value);//Insi*row*collom + Rowi*collom + Coli
	    	}//------------------------------------------------------
	    }
    }
    //MATRIX PRINTING:
    //============================================
    public void printAll(int ei) {printA(ei);printC(ei);printAT(ei);printB();}
    //------------------
     public void printAT(int ei)
    {
    	 System.out.println("At["+E[ei].ins()+" X "+E[ei].row()+" X "+E[ei].Anchor_Arowlayer(E[ei].rls()-1)+"]:\n================================");
    	 for(int Insi=0; Insi<E[ei].ins(); Insi++) 
    	 {
    		 System.out.println("At[("+Insi+") X "+E[ei].row()+" X "+E[ei].Anchor_Arowlayer(E[ei].rls()-1)+"]:\n-----------------");
    		 for(int Comi = 0; Comi < common ; Comi++)
    		 {
    			System.out.print("->[Comi]=>["+Comi+"]:  ");
		    	int anchbase;
		    	if(Insi==0) {anchbase=0;}else {anchbase=E[ei].Anchor_Arowlayer(Insi-1);}
		    	int inrowsize = E[ei].Anchor_Arowlayer(Insi)-anchbase;
		    	//---------------
		    	int AoBcomp_anch_p;// Setting anchorpoints:
		    	if(Insi!=0) {AoBcomp_anch_p=E[ei].Anchor_AoBcomp(Insi-1);}else {AoBcomp_anch_p=0;}
		    	//-------------------------
		    	int Bcomp_i=0;
		    	int Acomi=0;
		    	int Bcomi=0;
		    	//Sub array iteration (connection phase discriptor)
		    	while(AoBcomp_anch_p<(E[ei].Anchor_AoBcomp(Insi))) // p iterating to the next subarray!
		    	{
		  			//################################// This is within one row where Ii & Ri is given!
		    		if(0<E[ei].AoBcomp(AoBcomp_anch_p)) 
		    		{//::::::::::::::::::::::
		    			int anchor = Bcomp_i+E[ei].AoBcomp(AoBcomp_anch_p);
		    			while(Bcomp_i<anchor) 
		    			{ 
		    				int subanchor = Bcomi+Bcomp[Bcomp_i];
		    				while(Bcomi<subanchor) 
		    				{
		    					if(Bcomi==Comi)//This is the invers procedure: com*col threading: This already loops through comi! -> checking if Bcmoi==Comi!!...also->some rows of A-transposed might be empty! 
		    					{
		    						for(int Rowi=0; Rowi<E[ei].row(); Rowi++)
		    						{
		    							System.out.print("["+E[ei].matA(Acomi + inrowsize*Rowi + anchbase*E[ei].row())+"]");//...row*Insi
		    						}
		    					}
		    					Bcomi++; Acomi++;
		    				}
		    				Bcomp_i++;
		    			}
		    		}//::::::::::::::::::::::
		    		else//======================>>>>  A_comp contains within itself A configuration using B config within Bcomp
		    		{//::::::::::::::::::::::
		    			int anchor = Bcomp_i+E[ei].AoBcomp(AoBcomp_anch_p)*-1;
		    			while(Bcomp_i<anchor) 
		    			{
		    				int subanchor = Bcomi+Bcomp[Bcomp_i];
		    				while(Bcomi<subanchor) 
		    				{
		    					if(Bcomi==Comi)//This is the invers procedure: com*col threading: This already loops through comi! -> checking if Bcmoi==Comi!!...also->some rows of A-transposed might be empty! 
		    					{
		    						for(int Rowi=0; Rowi<E[ei].row(); Rowi++)
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
    public void printA(int ei)
    {
    	System.out.println("A["+E[ei].rls()+" X "+E[ei].row()+" X "+E[ei].Anchor_Arowlayer(E[ei].rls()-1)+"]:\n================================");
    	for(int Insi=0; Insi<E[ei].ins(); Insi++) 
    	{
    		System.out.println("A[("+Insi+") X "+E[ei].row()+" X "+E[ei].Anchor_Arowlayer(E[ei].rls()-1)+"]:\n-----------------");

    		for(int Rowi = 0; Rowi < E[ei].row() ; Rowi++)
    		{
    			System.out.print("->[Rowi]=>["+Rowi+"]:  ");
    			int anchbase;
    	    	if(Insi==0) {anchbase=0;}else {anchbase=E[ei].Anchor_Arowlayer(Insi-1);}
    	    	int inrowsize = E[ei].Anchor_Arowlayer(Insi)-anchbase;
    			//---------------
    			int AoBcomp_anch_p;// Setting anchor points:
    			if(Insi!=0) {AoBcomp_anch_p=E[ei].Anchor_AoBcomp(Insi-1);}else{AoBcomp_anch_p=0;}
    			//-------------------------
    			int Acomi=0;
    			int Bcomi=0;
    			int Bcomp_i=0;
    			//Sub array iteration (connection phase discriptor)
    			while(AoBcomp_anch_p<(E[ei].Anchor_AoBcomp(Insi))) // p iterating to the next sub-array!
    			{
    				//################################// This is within one row where Ii & Ri is given!
    				if(0<E[ei].AoBcomp(AoBcomp_anch_p)) 
    				{//::::::::::::::::::::::
    					int anchor = Bcomp_i+E[ei].AoBcomp(AoBcomp_anch_p);
    					while(Bcomp_i<anchor) 
    					{
    						int subanchor = Bcomi+Bcomp[Bcomp_i];
    						while(Bcomi<subanchor) 
    						{
    			    			System.out.print("["+E[ei].matA(Acomi + Rowi*inrowsize+anchbase*E[ei].row())+"]");
    							Bcomi++; Acomi++;
    						}
    						Bcomp_i++;
    					}
    				}//::::::::::::::::::::::
    				else//======================>>>>  A_comp contains within itself A configuration using B config within Bcomp
    				{//::::::::::::::::::::::
    					int anchor = Bcomp_i+E[ei].AoBcomp(AoBcomp_anch_p)*-1;
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
    public void printC(int ei) 
    {
    	System.out.println("C["+E[ei].ins()+" X "+E[ei].row()+" X "+collom+"]:\n================================");
    	for(int Insi=0; Insi<E[ei].ins(); Insi++) 
    	{
    		System.out.print("C[("+Insi+") X "+E[ei].row()+" X "+collom+"]:\n-----------------");
    		for(int Ci = 0; Ci < E[ei].row() * collom ; Ci++)
    		{
    			if(Ci%collom==0) {System.out.print("\n->[Rowi]=>["+Ci/collom+"]:  ");}///collom
    			System.out.print("["+E[ei].C(Insi*E[ei].row()*collom + ((int)(((Ci)/(collom)))) + E[ei].row()*((int)((Ci%collom))))+"]");     
    		}
    		System.out.println('\n');
    	}
    }
  //==========================================================================
    
    //Comparing CPU C-Matrix with with GPU C-Matrix!
    public void compareResults(int ei)
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
    		for(int Ci = 0; Ci < E[ei].row() * collom * E[ei].ins(); Ci++)
    		{
    			if(Math.abs(E[ei].matC(Ci)-E[ei].C(Ci))>0.001)
    			{
    				System.out.println("C:"+E[ei].C(Ci)+"; matC:"+E[ei].matC(Ci)+";");
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
