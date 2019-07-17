package neureka.unit;

import neureka.main.core.NVNode;

import neureka.utility.NMessageFrame;
import java.math.BigInteger;

public class NVTesting {

    protected NMessageFrame Console;
    protected NMessageFrame ResultConsole;
    protected String bar = "[|]";
    protected String line = "--------------------------------------------------------------------------------------------";

    private int success = 0;
    private int tests = 0;
    private String session = "";

    protected NVTesting(NMessageFrame console, NMessageFrame resultConsole) {
        Console = console;
        ResultConsole = resultConsole;
    }

    protected void performForwardBackwardOn(NVNode[] Structure){
        for(int i=0; i<Structure.length; i++) {Structure[i].asExecutable().loadLatest();}
        for(int i=0; i<Structure.length; i++) {Structure[i].asExecutable().forward();}
        for(int i=0; i<Structure.length; i++) {Structure[i].asExecutable().loadLatest();  }
        for(int i=0; i<Structure.length; i++) {Structure[i].asExecutable().loadTrainableState(BigInteger.ZERO);}
        for(int i=0; i<Structure.length; i++) {Structure[i].asExecutable().Backward(BigInteger.ZERO);}
    }

    protected void printNetStructView(NVNode[] Structure, NVNode head){
        for(int i=0; i<Structure.length; i++) {
            String expression = "null";
            if(Structure[i].asCore().getFunction()!=null) {
                expression = Structure[i].asCore().getFunction().toString();
            }
            Console.print(bar+"  Root ["+i+"]: ");
            if(Structure[i]==head) {Console.println(expression+"  => (head)");}
            else {
                Console.println(expression);
            }
        }
    }

    protected void printSessionStart(String message){
        session = "";
        success = 0;
        tests = 0;
        Console.println("");
        Console.println(bar+"  "+message);
        Console.println("[O][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=]=>");
        Console.println(bar+line);
        printlnResult("");
        printlnResult(bar+"  "+message);
        printlnResult("[O][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=]=>");
        printlnResult(bar);
    }
    protected int printSessionEnd(){
        Console.println(bar+"  "+((success>0)?"test successful!"+" "+success:"test failed!"+" "+(tests+success))+"/"+tests);
        Console.println("[O][=][=][=][=][=][=][=][=][=][=][=]|>");
        ResultConsole.println(bar+"  "+((success>0)?"test successful!"+" "+success:"test failed!"+" "+(tests+success))+"/"+tests);
        ResultConsole.println("[O][=][=][=][=][=][=][=][=][=][=][=]|>");
        return success;
    }

    protected boolean assertEqual(double result, double expected){
        tests++;
        if(result==expected) {
            println(bar+"  [result]:("+result+") == [expected]:("+expected+") -> test successful.");
            success = (success<0)?success:success+1;
            println(bar+line);
            return true;
        } else {
            println(bar+"  [result]:("+result+") =|= -> test failed!");
            success = (success<0)?success-1:-1;
            println(bar+line);
            return false;
        }
    }

    protected boolean assertEqual(String name, String result, String expected){
        tests++;
        if(result.equals(expected)) {
            println(bar+"  ["+name+"]:("+result+") "+((result.length()>22)?"\n"+bar+"    ==   -> test successful!"+"\n"+bar+" ":"==")+" [expected]:("+expected+")"+((result.length()>22)?"":" -> test successful!"));
            success = (success<0)?success:success+1;
            println(bar+line);
            return true;
        } else {
            println(bar+"  ["+name+"]:("+result+") "
                    +((result.length()>22)?"\n"+bar+"    =|=   -> test failed!"+"\n"+bar+" ":"=|=")+" [expected]:("+expected+")"+((result.length()>22)?"":" -> test failed!"));
            success = (success<0)?success-1:-1;
            println(bar+line);
            return false;
        }
    }

    protected boolean assertEqual(String result, String expected){
        tests++;
        if(result.equals(expected)) {
            println(bar+"  [result]:("+result+") "+((result.length()>22)?"\n"+bar+"    ==  "+"\n"+bar+" ":"==")+" [expected]:("+expected+") -> test successful.");
            success = (success<0)?success:success+1;
            println(bar+line);
            return true;
        } else {
            println(bar+"  [result]:("+result+") "+((result.length()>22)?"\n"+bar+"    =|=  "+"\n"+bar+" ":"=|=")+" [expected]:("+expected+") -> test failed!");
            success = (success<0)?success-1:-1;
            println(bar+line);
            return false;
        }
    }

    protected String stringified(int[] a){
        String result = "";
        for(int ai : a) {
            result += ai+", ";
        }
        return result;
    }
    protected String stringified(short[] a){
        String result = "";
        for(short ai : a) {
            result += ai+", ";
        }
        return result;
    }
    public String stringified(double[] a){
        String result = "";
        for(double ai : a) {
            result += ai+", ";
        }
        return result;
    }
    protected String stringified(double[][] a){
        String result = "";
        for(double[] ai : a) {
            result+="(";
            for(double aii : ai){
                result += aii+", ";
            }
            result+="), ";
        }
        return result;
    }
    protected String stringified(float[] a){
        String result = "";
        for(float ai : a) {
            result += ai+", ";
        }
        return result;
    }

    protected void print(String message){
        session+=message;
        Console.print(message);
    }
    protected void println(String message){
        session+=message+"\n";
        Console.println(message);
    }
    protected void printResult(String message){
        ResultConsole.print(message);
    }
    protected void printlnResult(String message){
        ResultConsole.println(message);
    }

}
