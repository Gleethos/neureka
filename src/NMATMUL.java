import com.aparapi.Kernel;
import com.aparapi.device.Device;


public class NMATMUL {

    @SuppressWarnings("deprecation")
	public static void main(String [] args) throws Exception
    {
        final int r = 1024;//1024;//3
        final int com = 1024;//r;//4;//4
        final int col = 1024;//r;//5;//5
        AparapiMatMul ap = new AparapiMatMul(r, com, col);
        //ap.printAll();
        Device device = Device.best();
    	//com.aparapi.Range range = device.createRange2D(r,col);
    	com.aparapi.Range range = device.createRange(r*com*col);
        try 
        {
        	ap.setExecutionMode(Kernel.EXECUTION_MODE.GPU);
        	long  time1 = System.currentTimeMillis();
        	ap.execute(range);
        	System.out.println("Time taken for kenel execution in "+ ap.getExecutionMode()+" mode is :"+(System.currentTimeMillis() - time1));
        }	
        catch(NullPointerException ne)
        {
            ne.printStackTrace();
        }
        //ap.printResults();
        long time1 = System.currentTimeMillis();
        //ap.cpuMatMulCalc();
        System.out.println("Time taken for kenel execution in Sequential CPU mode is :"+ (System.currentTimeMillis() - time1));
        //ap.printAll();
        ap.compareResults();
        //ap.printResults();
        ap.dispose();
    }
}
 
class AparapiMatMul extends Kernel {
 
	//@TFConstant
    float matA[];
    float matB[];
    float matC[];
    float C[];
    float B[];
    
    int row ;
    int common;
    int collom;
 
    @Override
    public void run() // A x B = C
    {
    	invers();
    	/*
    	int Coli = getGlobalId() % collom;
        int Rowi = (getGlobalId()-Coli) /collom;
        
        float _value = 0;
        for(int Comi = 0; Comi < common; Comi++)
        {
            _value += matA[Rowi * common + Comi] * matB[Comi * collom + Coli];
        }
        matC[Rowi * collom + Coli] = _value;
		*/
    }
    
    public void invers() //At x C = B
    {
    	int Coli = getGlobalId() % collom;
        int Comi = (getGlobalId()-Coli) /collom;
        
        float value = 0;
        for(int Rowi = 0; Rowi < row; Rowi++)
        {
            //_value += matA[Ri * common + Comi] * matB[Comi * collom + Coli];
        	value += matA[Comi + Rowi*common] * matC[Rowi * collom + Coli];//_value += matA[Comi * row + Rowi] * matC[Rowi * collom + Coli];
        }
        //matC[Ri * collom + Coli] = _value;
        matB[Comi * collom + Coli] = value; // Seems right!
    }
    
    
    public AparapiMatMul(int row, int common, int collom)
    {
        this.row = row;
        this.common = common;
        this.collom = collom;
        matA = new float [common *    row];
        matB = new float [common * collom];
        matC = new float [row    * collom];
        
        //matA = new float [row    * common];
        //matC = new float [row    * collom];
        //matB = new float [common * collom];
        B = new float[common * collom];
        C = new float[row * collom];
        //matC should be initialized with zeros 
        for(int Ri = 0; Ri < row; Ri++ )
        {
            for(int Coli = 0 ; Coli < collom; Coli++ )
            {
                matC[Ri * collom + Coli ] = Ri * collom + Coli;
                C[Ri * collom + Coli] = Ri * collom + Coli;
            }
        }
        //Here matrix A is initialized with random number
        for(int Ri = 0; Ri < row; Ri++ )//CORRECT! A: row X com
        {
            for(int Comi = 0 ; Comi < common; Comi++ )
            {
                matA[Ri * common +Comi] = Ri * common+Comi;// new Random().nextFloat();
            }
        }
        // Here matrix B is initialized with random numbers
        for(int Comi = 0; Comi < common; Comi++ )// B: com X col
        {
            for(int Coli = 0 ; Coli < collom; Coli++ )
            {
                matB[Comi * collom + Coli] = Comi * collom + Coli;//new Random().nextFloat();
                B[Comi * collom + Coli] = Comi * collom + Coli;
            }
        }
    }
 
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
 
    public void cpuMatMulCalc()
    {
    	cpuMatMulCalcT();
    	/*
     System.out.println("\nSequential Execution on CPU");
         for(int Ri = 0;Ri < row; Ri++)
            {
                for(int Coli = 0; Coli < collom; Coli++)
                {
                    float sum = 0;
                    for(int Comi = 0; Comi < common; Comi++)
                    {
                        sum += matA[Ri*common+Comi] * matB[Comi*collom+Coli];
                    }
                    C[Ri * collom + Coli] = sum;
                }
 
            }
            */
    }
    public void cpuMatMulCalcT()
    {
     System.out.println("\nSequential Execution on CPU");
         for(int Comi = 0; Comi < common; Comi++)
            {
                for(int Coli = 0; Coli < collom; Coli++)
                {
                    float sum = 0;
                    for(int Romi = 0; Romi < row; Romi++)
                    {
                        sum += matA[Comi+common*Romi] * matC[Romi*collom+Coli];
                    }
                    B[Comi * collom + Coli] = sum;
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
  //============================================
    
    //Comparing CPU C-Matrix with with TDevice C-Matrix!
    public void compareResults()
    {
        boolean equal = true;
        for(int Ci = 0; Ci < row * collom ; Ci++)
        {
        	System.out.println("C:"+C[Ci]+"; matC:"+matC[Ci]+";");
            if(Math.abs(matC[Ci]-C[Ci])>0.00000)
            {
                equal = false;// break;
            }
        }
        if(equal==false) {System.out.println("Results are not equal");}
        else	   {System.out.println("Results are equal.. Tested thoroughly!!!");}
        
        compareResultsT();
    }
    public void compareResultsT()
    {
        boolean equal = true;
        for(int Ci = 0; Ci < row * collom ; Ci++)
        {
        	//System.out.println("B:"+B[Ci]+"; matB:"+matB[Ci]+";");
            if(Math.abs(matC[Ci]-C[Ci])>0.00000)
            {
                equal = false;// break;
            }
        }
        if(equal==false) {System.out.println("Results are not equal");}
        else	   {System.out.println("Results are equal.. Tested thoroughly!!!");}
    }
}
