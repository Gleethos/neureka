
public class NGene {

	private static String CREATE = "EXP";
	
	private static String CONNECT = "CON";
	
	private static String SUPRESS = "SUP";
	
	private static String ACTIVATE = "ACT";
	
	private static String INVERT = "INV";
	
	private static String JUMP = "JMP";

	//====================================================================================
	private static String GENEFORMAT = 
			"G[id,mutatable,toBeMutable,size,gradientStatic]:[EXP,CON,SUP[...]]";
	
	private static String EXPFORMAT =
			"EXP[id]:[I[CON[id]:[W[1,23,3,5,32]]],I[CON[id]:[W[8,9,6]]],I[],I[]]";
	
	private static String CONNECTFORMAT =
			"CON[id]:[W[4,5,6,3,4,5]]";
	
	private static String HEAD = "M((mI0*(mI0*mI1+4*mI2)),[mI0:[],])";
	
}
