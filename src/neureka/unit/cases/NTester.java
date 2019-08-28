package neureka.unit.cases;

import neureka.frame.NMessageFrame;

public class NTester {

    protected NMessageFrame Console;
    protected NMessageFrame ResultConsole;
    protected String bar = "[|]";
    protected String line = "--------------------------------------------------------------------------------------------";
    private int success = 0;
    private int tests = 0;
    private String session = "";

    protected NTester(String name) {
        this.Console = new NMessageFrame(name+" - TEST PROCESS");
        this.ResultConsole = new NMessageFrame(name+" - TEST RESULT");
    }

    protected void printSessionStart(String message){
        this.session = "";
        this.success = 0;
        this.tests = 0;
        this.Console.println("");
        this.Console.println(bar+"  "+message);
        this.Console.println("[O][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=]=>");
        this.Console.println(bar+line);
        printlnResult("");
        printlnResult(bar+"  "+message);
        printlnResult("[O][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=]=>");
        printlnResult(bar);
    }
    protected int printSessionEnd(){
        this.Console.println(bar+"  "+((success>0)?"test successful!"+" "+success:"test failed!"+" "+(tests+success))+"/"+tests);
        this.Console.println("[O][=][=][=][=][=][=][=][=][=][=][=]|>");
        this.ResultConsole.println(bar+"  "+((success>0)?"test successful!"+" "+success:"test failed!"+" "+(tests+success))+"/"+tests);
        this.ResultConsole.println("[O][=][=][=][=][=][=][=][=][=][=][=]|>");
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
            println(bar+"  [result]:("+result+") =|= [expected]:("+expected+") -> test failed!");
            success = (success<0)?success-1:-1;
            println(bar+line);
            return false;
        }
    }

    protected boolean assertContains(String name, String result, String expected){
        tests++;
        if(result.contains(expected)) {
            println(bar+"  ["+name+"]:("+result+") "+((result.length()>22)?"\n"+bar+"    contains   -> test successful!"+"\n"+bar+" ":"contains")+" [expected]:("+expected+")"+((result.length()>22)?"":" -> test successful!"));
            success = (success<0)?success:success+1;
            println(bar+line);
            return true;
        } else {
            println(bar+"  ["+name+"]:("+result+") "
                +((result.length()>22)
                    ?"\n"+bar+"    not contains   -> test failed!"+"\n"+bar+" "
                    :"not contains")+" [expected]:("+expected+")"+((result.length()>22)?"":" -> test failed!"));
            success = (success<0)?success-1:-1;
            println(bar+line);
            return false;
        }
    }

    protected boolean assertEqual(String name, String result, String expected){
        tests++;
        if(result.equals(expected)) {
            println(bar+"  ["+name+"]:("+result+") "+((result.length()>22)
                    ?"\n"+bar+"    ==   -> test successful!"+"\n"+bar+" ":"==")+" [expected]:("+expected+")"+((result.length()>22)?""
                    :" -> test successful!"));
            success = (success<0)?success:success+1;
            println(bar+line);
            return true;
        } else {
            println(bar+"  ["+name+"]:("+result+") "
                    +((result.length()>22)
                    ?"\n"+bar+"    =|=   -> test failed!"+"\n"+bar+" "
                    :"=|=")+" [expected]:("+expected+")"+((result.length()>22)?"":" -> test failed!"));
            success = (success<0)?success-1:-1;
            println(bar+line);
            return false;
        }
    }

    protected boolean assertEqual(String result, String expected){
        tests++;
        if(result.equals(expected)) {
            println(bar+"  [result]:("+result+") "+((result.length()>22)
                    ?"\n"+bar+"    ==  "+"\n"+bar+" "
                    :"==")+" [expected]:("+expected+") -> test successful.");
            success = (success<0)?success:success+1;
            println(bar+line);
            return true;
        } else {
            println(bar+"  [result]:("+result+") "+((result.length()>22)
                    ?"\n"+bar+"    =|=  "+"\n"+bar+" "
                    :"=|=")+" [expected]:("+expected+") -> test failed!");
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
    public void println(String message){
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
