package neureka.main.exec.gpu;

public class Concept {

	
	public Concept() {
		
	}
	
	public static void main(String[] args)
	{
		
		String func = "(i3+i5)*i2-(i2*i3*i4-(33/i2))";
		for(int i=0; i<func.length(); i++) {
			System.out.println(Integer.toString(func.charAt(i)));
		}
		//to do
		// creating char map, 
	}
	
	
	//given:
	int inpsize;
	
	
	
	//---------------
	
	
	int[] NodeData_anchor; // -> negative value mean that target package is advanced!
	// advanced means that there is a 'static' output! 
	// memory nodes within root astructures have such kind of outputs!
	//=> NEGATIVE VALUES ARE MEMORY NODES => have extra output within NodeData;
	//... this array is a sort of pointer structure...
	//... input sizr is calculated from it:
	// if current pointer is positive -> (|p[curIndex]|-p[currIndex-1])/2
	//else : (|p[curIndex]|-|p[currIndex-1]|)/2
	
	int[] NodeFid; // size ==> number of nodes;
	
	byte[] FuncData;//-> char map used to interprete...
	//algorithm uses stack to calc output.
	/*What is being calvulated?
	 * 
	 * -> func output
	 * 
	 * -> derivatives.
	 * 
	 * The alglgorithm calculates th derivative for every input independently.
	 * 
	 * lets take (i0 + 3)* i1 - (i0 * i2)
	 * 
	 * alg goes from left to right and dives into
	 * the deepest bracket block first!
	 * -> the bracket result is stored on stack when moving to next component
	 * within current bracket!
	 * (Klapustri rule is considered bracket scope aswell!!!!!!!!!!!!)
	 * meaning:   (i0 * i3 * 3 + 5 + i2)
	 * is really: ((i0 * i3 * 3) + 5 + i2)
	 * 
	 * after all the brackets are solved the current statement is analyzed!
	 * 
	 * + -> return 0 if derivative target is not directly or indirectly included
	 * 
	 * + -> return c1' + c2 ... c3; 
	 * 
	 * */
	
	
	/*Ok,
	 * consider this:
	 * 
	 * This whole algorithm needs one big of array!
	 * float[] of -> C matrix of Matmul;
	 * 
	 * then there is a big array of nodes!
	 * some of these nodes are head nodes!
	 * 
	 * meaning: -> they derive the of array!!
	 * ... in practice there isn't always 
	 * a derivative for every head->of relationship (because they are not connected....)
	 * 
	 * ...but still:
	 * the algorithm needs that freedom in order to stay simple!
	 * 
	 * every node has a local head to input relationship
	 * ...and (possibly) a head to of relationship!
	 * 
	 * 
	 * */
	
	int[]  FuncData_anchor; // FuncData_anchor.length == NodeFid.length
	
	byte[] NodeIsSrc;// NodeIsSrc.length == NodeFid.length == FuncData_anchor.length
	// A source node can either be part of matmul matrix C or additional input.
	
	float[] StructData; 
	// contains activation outputs of memory cells, biases and weights...
	// Target pointer are contained within FunctionData array.
	int[] StructData_anchor;
	
	int[] ConnSpace; // Format: [payloadsize_n1][insize][targ1][targ2]...[payloadsize_n2][...]...
	// which func node is connected to what?
	// -> positive targ -> other nodes
	// -> negative targ -> of nodes
	
	float[] Result;
	//size => headnodecount * out size;
	
	public void run() 
	{	
		float[] NodeData; // segregated into Out,(old Out), Inp, dInp
		//exists only within kernel run!!! -> instantiated locally! 
		NodeData = new float[NodeData_anchor[NodeData_anchor.length-1]];	
		
		
	}
	
	public void createFuncCode(String funtion) {
		
		//(1+4*74)/2-3-1
		//=> [9][1][.][+][4][.][*][74][.][/]
		
		
		
	}
	
	public void activate(int[] func, float[] data) {
		
		
		
	}
	
	public void derive(int[] func, float[] data, int index){
		
	}
	
	
	
	
	
	
	
	
	
	
	
}
