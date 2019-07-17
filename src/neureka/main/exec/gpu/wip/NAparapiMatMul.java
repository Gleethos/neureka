package neureka.main.exec.gpu.wip;

import java.util.Random;

import com.aparapi.Kernel;

public class NAparapiMatMul extends Kernel 
{
	private float matA[];
	private float matB[];
	private float matC[];
	private float C[];
	private float B[];
    
    @Constant int row ;
    @Constant int common;
    @Constant int collom;
    @Constant boolean[] inversed = {true};
  //====================================================
    public int convectionSize() {if(inversed[0]) {return collom*common;}else{return row*collom;}}
    
    @Override
    public void run() 
    {
    	int Coli = getGlobalId() % collom;
    	
    	if(inversed[0]) //At x C = B
    	{//------------------------------------------------------
    		int Comi = (getGlobalId()-Coli) /collom;
        
    		float value = 0;
    		for(int Rowi = 0; Rowi < row; Rowi++)
    		{
    			value += matA[Rowi * common + Comi] * matC[Rowi * collom + Coli];
    		}
    		matB[Comi * collom + Coli] = value;
    	}//------------------------------------------------------
    	else // A x B = C
    	{//------------------------------------------------------
    		int Rowi = (getGlobalId()-Coli) /collom;
        
    		float value = 0;
    		for(int Comi = 0; Comi < common; Comi++)
    		{
    			value += matA[Rowi * common + Comi] * matB[Comi * collom + Coli];
    		}
    		matC[Rowi * collom + Coli] = value;
    	}//------------------------------------------------------
    }
  //==========================================================================
    public void cpuMatMulCalc()
    {
    	if(inversed[0]) 
    	{
    		 System.out.println("\nSequential Execution on CPU");
    		 for(int Comi = 0; Comi < common; Comi++)
            {
                for(int Coli = 0; Coli < collom; Coli++)
                {
                    float sum = 0;
                    for(int Rowi = 0; Rowi < row; Rowi++)
                    {
                        sum += matA[common*Rowi+Comi] * matC[Rowi*collom+Coli];
                    }
                    B[Comi * collom + Coli] = sum;
                }
            }
    	}
    	else
    	{
    		System.out.println("\nSequential Execution on CPU");
    		for(int Rowi = 0;Rowi < row; Rowi++)
            {
                for(int Coli = 0; Coli < collom; Coli++)
                {
                    float sum = 0;
                    for(int Comi = 0; Comi < common; Comi++)
                    {
                        sum += matA[Rowi*common+Comi] * matB[Comi*collom+Coli];
                    }
                    C[Rowi * collom + Coli] = sum;
                }
 
            }
    	}
    }
    //==========================================================================
    
    public void setA(float[] newA) {matA=newA;}
    public void setB(float[] newB) {matB=newB;}
    public void setC(float[] newC) {matC=newC;}
    
    public float[] getA() {return matA;}
    public float[] getB() {return matB;}
    public float[] getC() {return matC;}
    
    public void setTestB(float[] newB) {B=newB;}
    public void setTestC(float[] newC) {C=newC;}
    
    public float[] getTestB() {return B;}
    public float[] getTestC() {return C;}
    

    public NAparapiMatMul(int row, int common, int collom)
    {
        this.row = row;
        this.common = common;
        this.collom = collom;
        matA = new float [common *    row];
        matB = new float [common * collom];
        matC = new float [row    * collom];
        
        B = new float[common * collom];
        C = new float[row * collom];

        //Here matrix A is initialized with random number
        for(int Ri = 0; Ri < row; Ri++ )
        {
            for(int Comi = 0 ; Comi < common; Comi++ )
            {
            	float f = new Random().nextFloat();
                matA[Ri * common +Comi] = (Ri * common+Comi)%11-5;// new Random().nextFloat();
            }
        }
        //matC should be initialized with zeros 
        for(int Ri = 0; Ri < row; Ri++ )
        {
            for(int Coli = 0 ; Coli < collom; Coli++ )
            {
            	float f = new Random().nextFloat();
                matC[Ri * collom + Coli ] = (Ri * collom + Coli)%11-5;
                C[Ri * collom + Coli] = (Ri * collom + Coli)%11-5;
            }
        }
        // Here matrix B is initialized with random numbers
        for(int Comi = 0; Comi < common; Comi++ )// B: com X col
        {
            for(int Coli = 0 ; Coli < collom; Coli++ )
            {
            	float f = new Random().nextFloat();
                matB[Comi * collom + Coli] = (Comi * collom + Coli)%11-5;//
                B[Comi * collom + Coli] = (Comi * collom + Coli)%11-5;
            }
        }
    }
   //==========================================================================
    public void printResults()
    {
        for(int Ri = 0; Ri < row; Ri++ )
        {
            for(int C2i = 0 ; C2i < collom; C2i++ )
            {
                System.out.print(matC[Ri * collom + C2i]+"    ");
            }
        }
    }
    //MATRIX PRINTING:
    //============================================
    public void printAll() {printAT();printA();printB();printC();}
    //------------------
     public void printAT()
    {System.out.print("AT["+row+" X "+common+"]:");
        for(int Ai = 0; Ai < common ; Ai++)
        {System.out.println();
         for(int Aii=0; Aii<row; Aii++) {System.out.print("["+matA[(Aii*common)+Ai]+"]");}
        }
     System.out.println('\n');
    }
   //------------------
    public void printA()
    {System.out.print("A["+row+" X "+common+"]:");
        for(int Ai = 0; Ai < row * common ; Ai++)
        {if(Ai%common==0) {System.out.println();}
         System.out.print("["+matA[Ai]+"]");
        }
     System.out.println('\n');
    }
    public void printB()
    {System.out.print("B["+common+" X "+collom+"]:");
        for(int Bi = 0; Bi < common * collom ; Bi++)
        {if(Bi%collom==0) {System.out.println();}
         System.out.print("["+matB[Bi]+"]");
        }
     System.out.println('\n');
    }
    public void printC() {
     System.out.print("C["+row+" X "+collom+"]:");
    	for(int Ci = 0; Ci < row * collom ; Ci++)
        {if(Ci%collom==0) {System.out.println();}
         System.out.print("["+matC[Ci]+"]");     
        }
    System.out.println('\n');
    }
  //==========================================================================
    
    //Comparing CPU C-Matrix with with TDevice C-Matrix!
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
    		if(equal==false) {System.out.println("Results are not equal");}
    		else	  		 {System.out.println("Results are equal.. Tested thoroughly!!!");}
    	}
    	else
    	{
    		boolean equal = true;
    		for(int Ci = 0; Ci < row * collom ; Ci++)
    		{
    			if(Math.abs(matC[Ci]-C[Ci])>0.001)
    			{
    				System.out.println("C:"+C[Ci]+"; matC:"+matC[Ci]+";");
    				equal = false; break;
    			}
    		}
    		if(equal==false) {System.out.println("Results are not equal");}
    		else	  		 {System.out.println("Results are equal.. Tested thoroughly!!!");}
    	}
    }
      
}//##############################################################################################################


