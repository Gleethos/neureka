package neureka.core.modul.calc.fcomp.util;

import neureka.core.T;
import neureka.core.modul.calc.Device;
import neureka.core.modul.calc.FunctionFactory;
import neureka.core.modul.calc.GraphNode;
import neureka.core.modul.calc.fcomp.Constant;
import neureka.core.modul.calc.fcomp.Function;

import java.util.ArrayList;

public class Calculation {
    private interface Actor{ void apply(Integer i, double[] v1, double[] v2);}

    private static void foreach(T t1, T t2, Actor action){
        double[] inputValue = (t1.value()==null)?new double[t1.size()]:t1.value();
        double[] outputValue = (t2.value()==null)?new double[t2.size()]:t2.value();
        t1.foreach((i)->{ action.apply(i, inputValue, outputValue); });
        t2.setValue(outputValue);
        t1.setValue(inputValue);
    }
    //Activation stage 1: Function determination!
    //============================================================================================================================================================================================
    /**
     *  Responsible for handling functions with id's 0-9  (single input functions!)
     * */
    public static T tensorActivationOf(T input, int f_id, boolean derive, boolean tipReached, boolean isFlat) {
        T output = T.factory.newTensor(input.shape(), input.translation());
        if(!derive && !isFlat){//implies !tipReached ==true // only flat functions can be executed
            output.addModule(new GraphNode(output, new T[]{input}, f_id, true, false));
            return output;
        }
        if(input.isOutsourced()){
            Device device = (Device) input.find(Device.class);
            device.calculate(new T[]{output, input}, f_id, (derive)?0:-1);
        }else{
            foreach(input, output, (i, inputValue, outputValue)->{
                outputValue[i] = Calculation.scalarActivationOf(inputValue[i], f_id, derive);
            });
        }
        if(!derive){//&& tipReached
            output.addModule(new GraphNode(output, new T[]{input}, f_id, false, (isFlat)?(!tipReached)?true:false:false));
        }
        return output;
    }
    //TODO: j is not handled properly!!!
    /**
     *   Responsible for handling functions with multiple inputs!
     * */
    public static T tensorActivationOf(T[] input, int f_id, int j, int d, ArrayList<Function> Srcs, boolean tipReached, boolean isFlat) {
        T output = T.factory.newTensor(input[0].shape(), input[0].translation());
        /**
         *   Teh code below deals with deep functions (non flat):
         * */
        if(d<0 && !isFlat){//implies !tipReached ==true // only flat functions can be executed
            if(f_id<9){
                output.addModule(new GraphNode(output, input, Context.register[f_id]+"(I["+((j<0)?0:j)+"])", true, false));
                return output;
            }else{
                if(Context.register[f_id].length()!=1){
                    /**  SUMMATION, PI,
                     * */
                    T[] tsrs = new T[input.length];
                    for(int i=0; i<tsrs.length; i++){
                        tsrs[i] =  Srcs.get(0).activate(input, i);
                    }// THIS NEEDS TO BE SOLVED!!
                    output.addModule(new GraphNode(output, tsrs, Context.register[f_id]+"(I[j])", true, false));
                    return output;
                }else{
                    /**      +, -, x, *, %, ....
                     * */
                    String operation = (Context.register[f_id].length()>1)?Context.register[f_id]:"";
                    T[] tsrs = new T[Srcs.size()];
                    boolean constantFound = false;
                    T template = null;
                    for(int i=0; i<tsrs.length; i++){//constants need to be figured out!
                        if(Srcs.get(i) instanceof Constant){
                            tsrs[i] = null;
                            constantFound = true;
                        }else{
                            tsrs[i] = (j<0)?Srcs.get(i).activate(input):Srcs.get(i).activate(input, j);
                            template = tsrs[i];
                        }
                        operation += "I["+i+"]"+((i+1<tsrs.length)?Context.register[f_id]:"");
                    }
                    if(constantFound){
                        for(int i=0; i<tsrs.length; i++){
                            tsrs[i] = (tsrs[i] != null)
                                    ? tsrs[i]
                                    : (j<0)
                                    ?T.factory.newTensor(Srcs.get(i).activate(new double[]{}), template.shape())
                                    :T.factory.newTensor(Srcs.get(i).activate(new double[]{}, j), template.shape());
                        }
                    }
                    if(j<0){
                        output.internalize(new FunctionFactory().newBuild(operation).activate(tsrs));
                    }else{
                        output.internalize(new FunctionFactory().newBuild(operation).activate(tsrs, j));
                    }
                    //output.addModule(new GraphNode(output, tsrs, operation, true, false));
                    return output;
                }
            }
        }
        /**
         *  The following code is reached in flat functions only:
         * */
        Device device = (Device) input[0].find(Device.class);
        boolean onSameDevice = T.utility.shareGuestDevice(input);
        if(onSameDevice){
            if(device!=null){
                device.add(output);
            }
            for (int ti = 0; ti < input.length; ti++) {
                device = (Device) input[ti].find(Device.class);
                T[] tsrs = new T[1+input.length];
                tsrs[0]=output;
                for(int tii=1; tii<tsrs.length; tii++){
                    tsrs[tii]=input[tii-1];
                }
                device.calculate(tsrs, f_id, d);
            }
        }else{
            if(Context.register[f_id]=="x"){
                if(d<0){
                    output = T.factory.convolution(input[0], input[1]);
                }else{
                    if(d==0){
                        output = input[1];
                    }else{
                        output = input[0];
                    }
                }
            }else{
                double[] inp = new double[input.length];
                T finalOutput = output;
                output.foreach((i)->{
                    for (int ti = 0; ti < input.length; ti++) {
                        inp[ti] = input[ti].value()[i];
                    }
                    finalOutput.value()[i] = Calculation.scalarActivationOf(inp, f_id, j, d, Srcs);
                });
            }
        }
        if(d<0){//&& tipReached
            T[] tsrs = input;
            if(f_id==18){
                tsrs = new T[Srcs.size()];
                for(int i=0; i<tsrs.length; i++){
                    tsrs[i] = Srcs.get(i).activate(input);
                }
            }
            if(j<0){
                //output.internalize(new FunctionFactory().newBuild(f_id, tsrs.length, false).activate(tsrs));
            }else{
                //output.internalize(new FunctionFactory().newBuild(f_id, tsrs.length, false).activate(tsrs, j));
            }//TODO: resolve: boolean isFlat is set true always here! tip reached is the only metric!
            output.addModule(new GraphNode(output, tsrs, f_id, false, true));//(isFlat)?(!tipReached)?true:false:false));
            /**
             *   Its always flat! And tip is not reached!
             * */
        }
        return output;
    }
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    public static double scalarActivationOf(double input, int f_id, boolean derive) {
        switch (f_id) {
            case 0: return getReLuOf(       input, derive);
            case 1: return getSigmoidOf(    input, derive);
            case 2: return getTanhOf(       input, derive);
            case 3: return getQuadraticOf(  input, derive);
            case 4: return getLigmoidOf(    input, derive);
            case 5: return getLinearOf(     input, derive);
            case 6: return getGaussianOf(   input, derive);
            case 7: return getAbsoluteOf(   input, derive);
            case 8: return getSinusOf(      input, derive);
            case 9: return getCosinusOf(    input, derive);
            default: return input;
        }
    }
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    public static double scalarActivationOf(double[] input, int f_id, int j, int d, ArrayList<Function> Variable) {
        switch (f_id) {
            case 10: return (j<0)? getSummation(      input, d, Variable) : getSummation(      input, j, d, Variable);
            case 11: return (j<0)? getPI(             input, d, Variable) : getPI(             input, j, d, Variable);
            case 12: return (j<0)? getPowerOf(        input, d, Variable) : getPowerOf(        input, j, d, Variable);
            case 13: return (j<0)? getDivision(       input, d, Variable) : getDivision(       input, j, d, Variable);
            case 14: return (j<0)? getMultiplication( input, d, Variable) : getMultiplication( input, j, d, Variable);
            case 15: return (j<0)? getModulo(         input, d, Variable) : getModulo(         input, j, d, Variable);
            case 16: return (j<0)? getSubtraction(    input, d, Variable) : getSubtraction(    input, j, d, Variable);
            case 17: return (j<0)? getAddition(       input, d, Variable) : getAddition(       input, j, d, Variable);
            case 18: return (j<0)? getMultiplication( input, d, Variable) : getMultiplication( input, j, d, Variable);
            default: return 0;
        }
    }

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    //Activation stage 1: Activation type determination
    //============================================================================================================================================================================================
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    public static double getReLuOf(double input, boolean derive) {
        double output;
        if (!derive) {
            if (input >= 0) {
                output = (input + Context.shift) *Context.inclination;
            } else {
                output = (input + Context.shift) * Context.secondaryInclination;
            }
            return output;
        } else {
            if (input >= 0) {
                output =Context.inclination;
            } else {
                output = Context.secondaryInclination;
            }
            return output;
        }
    }
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    public static double getSigmoidOf(double input, boolean derive) {
        if (!derive) {
            return 1 / (1 + Math.pow(Math.E, (-(input + Context.shift) *Context.inclination)));
        } else {
            return Context.inclination * (Math.pow(Math.E, -(input + Context.shift) *Context.inclination)) / (Math.pow((1 + Math.pow(Math.E, -(input + Context.shift) *Context.inclination)), 2) + 2 * Math.pow(Math.E, -(input + Context.shift) *Context.inclination));
        }
    }
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    public static double getTanhOf(double input, boolean derive) {
        if (!derive) {
            return ((input + Context.shift) *Context.inclination) / Math.pow((1 + Math.pow(((input + Context.shift) *Context.inclination), 2)), 0.5);
        } else {
            return (1 - Math.pow(((input + Context.shift) / Math.pow((1 + Math.pow((input + Context.shift), 2)), 0.5)), 2)) *Context.inclination;
        }
    }
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    public static double getQuadraticOf(double input, boolean derive) {
        if (!derive) {
            return ((input + Context.shift) * (input + Context.shift) *Context.inclination);
        } else {
            return 2 * input *Context.inclination + 2 * Context.shift *Context.inclination;
        }
    }
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    public static double getLigmoidOf(double input, boolean derive) {
        if (!derive) {
            return (Context.inclination * (input + Context.shift) + (Math.log(Math.pow(Math.E, -(input + Context.shift) *Context.inclination) + 1) / Math.log(Math.E)));
        } else {
            return getSigmoidOf(input, false);
        }
    }
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    public static double getLinearOf(double input, boolean derive) {
        if (!derive) {
            return Context.inclination * (input + Context.shift);
        } else {
            return Context.inclination;
        }
    }
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    public static double getGaussianOf(double input, boolean derive) {
        if (!derive) {
            return Math.pow(Math.E, -Math.pow(Context.inclination * (input + Context.shift), 2));
        } else {
            return -2 * (Context.inclination * (input + Context.shift)) * Math.pow(Math.E, -Math.pow(Context.inclination * (input + Context.shift), 2));
        }
    }
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    public static double getAbsoluteOf(double input, boolean derive) {
        if (!derive) {
            return Math.abs(input);
        } else {
            return (input<0)?-1:1;
        }
    }
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    public static double getSinusOf(double input, boolean derive) {
        if (!derive) {
            return Math.sin(input);
        } else {
            return Math.cos(input);
        }
    }
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    public static double getCosinusOf(double input, boolean derive) {
        if (!derive) {
            return Math.cos(input);
        } else {
            return -Math.sin(input);
        }
    }
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    private static double getSummation(double[] input, int j, int d, ArrayList<Function> Variable){
        if(d<0) {
            double sum = 0;
            boolean nothingDone = true;
            for (int Ii = 0; Ii < input.length; Ii++) {
                sum += Variable.get(0).activate(input, Ii);
                nothingDone = false;
            }
            if (nothingDone) {
                return Variable.get(0).activate(input);
            }
            return sum;
        }else{
            return Variable.get(0).derive(input, d, j);
        }
    }
    private static double getSummation(double[] input, int d, ArrayList<Function> Variable){
        if(d<0) {
            double sum = 0;
            boolean nothingDone = true;
            for (int Ii = 0; Ii < input.length; Ii++) {
                sum += Variable.get(0).activate(input, Ii);
                nothingDone = false;
            }
            if (nothingDone) {
                return Variable.get(0).activate(input);
            }
            return sum;
        }else{
            return Variable.get(0).derive(input, d);
        }

    }
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    private static double getPI(double[] input, int j, int d, ArrayList<Function> Variable){
        if(d<0) {
            double prod = 1;
            boolean nothingDone = true;
            for (int Ii = 0; Ii < input.length; Ii++) {
                //if (sources.get(0).dependsOn(Ii)) {
                prod *= Variable.get(0).activate(input, Ii);
                nothingDone = false;
                //}
            }
            if (nothingDone) {
                return Variable.get(0).activate(input, j);
            }
            return prod;
        }else{
            double u, ud, v, vd;
            u = Variable.get(0).activate(input, 0);
            ud = Variable.get(0).derive(input, d, 0);
            System.out.println(ud);
            for (int ji = 1; ji < input.length; ji++) {
                v = Variable.get(0).activate(input, ji);
                vd = Variable.get(0).derive(input, d, ji);
                ud = u * vd + v * ud;
                u *= v;
            }
            return ud;
        }
    }
    private static double getPI(double[] input, int d, ArrayList<Function> Variable){
        if(d<0) {
            double prod = 1;
            boolean nothingDone = true;
            for (int Ii = 0; Ii < input.length; Ii++) {
                //if (sources.get(0).dependsOn(Ii)) {
                prod *= Variable.get(0).activate(input, Ii);
                nothingDone = false;
                //}
            }
            if (nothingDone) {
                return Variable.get(0).activate(input);
            }
            return prod;
        }else{
            double u, ud, v, vd;
            u = Variable.get(0).activate(input, 0);
            ud = Variable.get(0).derive(input, d, 0);
            System.out.println(ud);
            for (int j = 1; j < input.length; j++) {
                v = Variable.get(0).activate(input, j);
                vd = Variable.get(0).derive(input, d, j);
                ud = u * vd + v * ud;
                u *= v;
            }
            return ud;
        }
    }
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    private static double getPowerOf(double[] input, int j, int d, ArrayList<Function> Variable){
        if(d<0) {
            double result = Variable.get(0).activate(input, j);
            for (int Vi = 1; Vi < Variable.size(); Vi++) {
                final double current = Variable.get(Vi).activate(input, j);
                result = Math.pow(result, current);
            }
            return result;
        }else{
            // d/dx(f(x)^g(x))=
            //	f(x)^g(x) * d/dx(g(x)) * ln(f(x))
            //	+ f(x)^(g(x)-1) * g(x) * d/dx(f(x))
            double fg, dg, lnf;
            double f = Variable.get(0).activate(input, j);
            double df = Variable.get(0).derive(input, d, j);
            double g;
            for (int i = 0; i < Variable.size() - 2; i++) {
                g = Variable.get(i + 1).activate(input, j);
                fg = f * g;
                dg = Variable.get(i + 1).derive(input, d, j);
                lnf = Math.log(f);
                df = fg * dg * lnf + f * (g - 1) * g * df;
                f = Math.pow(f, g);
            }
            return df;
        }
    }
    private static double getPowerOf(double[] input, int d, ArrayList<Function> Variable){
        if(d<0) {
            double result = Variable.get(0).activate(input);
            for (int Vi = 1; Vi < Variable.size(); Vi++) {
                final double current = Variable.get(Vi).activate(input);
                result = Math.pow(result, current);
            }
            return result;
        }else{
            // d/dx(f(x)^g(x))=
            //f(x)^g(x) * d/dx( g(x) ) * ln( f(x) )
            //+ f(x)^( g(x)-1 ) * g(x) * d/dx( f(x) )
            double fg, dg, lnf;
            double f = Variable.get(0).activate(input);
            double df = Variable.get(0).derive(input, d);
            double g;
            for (int i = 0; i < Variable.size() - 2; i++) {
                g = Variable.get(i + 1).activate(input);
                fg = f * g;
                dg = Variable.get(i + 1).derive(input, d);
                lnf = Math.log(f);
                df = fg * dg * lnf + f * (g - 1) * g * df;
                f = Math.pow(f, g);
            }
            return df;
        }
    }
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    private static double getDivision(double[] input, int j, int d, ArrayList<Function> Variable){
        if(d<0) {
            double result = Variable.get(0).activate(input, j);
            for (int Vi = 1; Vi < Variable.size(); Vi++) {
                final double current = Variable.get(Vi).activate(input, j);
                result /= current;
            }
            return result;
        }else{
            double u, ud, v, vd;
            u = Variable.get(0).activate(input, j);
            ud = Variable.get(0).derive(input, d, j);
            for (int i = 0; i < Variable.size() - 1; i++) {
                v = Variable.get(i + 1).activate(input, j);
                vd = Variable.get(i + 1).derive(input, d, j);
                ud = (ud * v - u * vd) / Math.pow(v, 2);
                u /= v;
            }
            return ud;
        }
    }
    private static double getDivision(double[] input, int d, ArrayList<Function> Variable){
        if(d<0) {
            double result = Variable.get(0).activate(input);
            for (int Vi = 1; Vi < Variable.size(); Vi++) {
                final double current = Variable.get(Vi).activate(input);
                result /= current;
            }
            return result;
        }else{
            double derivative = 0;
            double tempVar = Variable.get(0).activate(input);
            derivative = Variable.get(0).derive(input, d);

            for (int i = 0; i < Variable.size() - 1; i++) {
                double u, ud, v, vd;
                v = Variable.get(i + 1).activate(input);
                vd = Variable.get(i + 1).derive(input, d);
                u = tempVar;
                ud = derivative;
                derivative = (ud * v - u * vd) / Math.pow(v, 2);
                tempVar /= v;
            }
            return derivative;
        }
    }
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    private static double getMultiplication(double[] input, int j, int d, ArrayList<Function> Variable){
        if(d<0) {
            double result = Variable.get(0).activate(input, j);
            for (int Vi = 1; Vi < Variable.size(); Vi++) {
                final double current = Variable.get(Vi).activate(input, j);
                result *= current;
            }
            return result;
        }else{
            double u, ud, v, vd;
            u = Variable.get(0).activate(input, j);
            ud = Variable.get(0).derive(input, d, j);

            for (int ji = 1; ji < Variable.size(); ji++) {
                v = Variable.get(ji).activate(input, j);
                vd = Variable.get(ji).derive(input, d, j);
                System.out.println("ud" + (u * vd + v * ud) + "=u" + u + "*vd" + vd + "+v" + v + "*ud" + ud);
                ud = u * vd + v * ud;
                u *= v;
            }
            System.out.println("* d: " + ud + "; j: " + j);
            return ud;
        }
    }
    private static double getMultiplication(double[] input, int d, ArrayList<Function> Variable){
        if(d<0) {
            double result = Variable.get(0).activate(input);
            for (int Vi = 1; Vi < Variable.size(); Vi++) {
                final double current = Variable.get(Vi).activate(input);
                result *= current;
            }
            return result;
        }else{
            double u, ud, v, vd;
            u = Variable.get(0).activate(input);
            ud = Variable.get(0).derive(input, d);

            for (int j = 1; j < Variable.size(); j++) {
                v = Variable.get(j).activate(input);
                vd = Variable.get(j).derive(input, d);

                ud = u * vd + v * ud;
                u *= v;//this step can be avoided (TODO optimize)
            }
            return ud;
        }
    }
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    private static double getModulo(double[] input, int j, int d, ArrayList<Function> Variable){
        if(d<0) {
            double result = Variable.get(0).activate(input, j);
            for (int Vi = 1; Vi < Variable.size(); Vi++) {
                final double current = Variable.get(Vi).activate(input, j);
                result %= current;
            }
            return result;
        }else{
            return Variable.get(0).derive(input, d, j);// j ?
        }
    }
    private static double getModulo(double[] input, int d, ArrayList<Function> Variable){
        if(d<0) {
            double result = Variable.get(0).activate(input);
            for (int Vi = 1; Vi < Variable.size(); Vi++) {
                final double current = Variable.get(Vi).activate(input);
                result %= current;
            }
            return result;
        }else{
            return Variable.get(0).derive(input, d);
        }
    }
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    private static double getSubtraction(double[] input, int j, int d, ArrayList<Function> Variable){
        if(d<0) {
            double result = Variable.get(0).activate(input, j);
            for (int Vi = 1; Vi < Variable.size(); Vi++) {
                final double current = Variable.get(Vi).activate(input, j);
                result -= current;
            }
            return result;
        }else{
            double derivative = 0;
            for (int i = 0; i < Variable.size(); ++i) {
                //if (sources.get(i).dependsOn(index)) {
                if (i == 0) {
                    derivative += Variable.get(i).derive(input, d, j);
                } else {
                    derivative -= Variable.get(i).derive(input, d, j);
                }
                //}
            }
            return derivative;
        }
    }
    private static double getSubtraction(double[] input, int d, ArrayList<Function> Variable){
        if(d<0) {
            double result = Variable.get(0).activate(input);
            for (int Vi = 1; Vi < Variable.size(); Vi++) {
                final double current = Variable.get(Vi).activate(input);
                result -= current;
            }
            return result;
        }else{
            double derivative = 0;
            for (int i = 0; i < Variable.size(); ++i) {
                //if (sources.get(i).dependsOn(index)) {
                if (i == 0) {
                    derivative += Variable.get(i).derive(input, d);
                } else {
                    derivative -= Variable.get(i).derive(input, d);
                }
                //}
            }
            return derivative;
        }
    }
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    private static double getAddition(double[] input, int j, int d, ArrayList<Function> Variable){
        if(d<0) {
            double result = Variable.get(0).activate(input, j);
            for (int Vi = 1; Vi < Variable.size(); Vi++) {
                final double current = Variable.get(Vi).activate(input, j);
                result += current;
            }
            return result;
        }else{
            double derivative = 0;
            for (int i = 0; i < Variable.size(); ++i) {
                derivative += Variable.get(i).derive(input, d, j);
            }
            return derivative;
        }
    }
    private static double getAddition(double[] input, int d, ArrayList<Function> Variable){
        if(d<0) {
            double result = Variable.get(0).activate(input);
            for (int Vi = 1; Vi < Variable.size(); Vi++) {
                final double current = Variable.get(Vi).activate(input);
                result += current;
            }
            return result;
        }else{
            double derivative = 0;
            for (int i = 0; i < Variable.size(); ++i) {
                //if (sources.get(i).dependsOn(index)) {
                derivative += Variable.get(i).derive(input, d);
                //}
            }
            return derivative;
        }
    }
}
