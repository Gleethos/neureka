package neureka.core.function.factory.material;

import neureka.core.T;
import neureka.core.function.autograd.TGraphLock;
import neureka.core.function.factory.worker.FBuilder;
import neureka.core.function.autograd.TGraphBuilder;
import neureka.core.device.TDevice;
import neureka.core.function.TFunction;

import java.util.ArrayList;

public abstract class TFTemplate implements TFunction {

    protected int f_id;
    protected boolean isFlat;
    protected boolean doAD;
    ArrayList<TFunction> Srcs;

    protected TFTemplate(int f_id, boolean isFlat, ArrayList<TFunction> Srcs, boolean doAD){
        this.f_id = f_id;
        this.isFlat = isFlat;
        this.Srcs = Srcs;
        this.doAD = doAD;
    }

    @Override
    public boolean isFlat(){
        return  this.isFlat;
    }

    @Override
    public int id() {
        return this.f_id;
    }

    @Override
    public String type() {
        return TFunction.F_CACHE.REGISTER[f_id];
    }

    @Override
    public TFunction newBuild(String expression){
        return FBuilder.newBuild(expression, true);
    }

    @Override
    public String toString() {
        String reconstructed = "";
        if (Srcs.size() == 1 && TFunction.F_CACHE.REGISTER[f_id].length() > 1) {
            String expression = Srcs.get(0).toString();
            if (expression.charAt(0) == '(' && expression.charAt(expression.length() - 1) == ')') {
                return TFunction.F_CACHE.REGISTER[f_id] + expression;
            }
            return TFunction.F_CACHE.REGISTER[f_id] + "(" + expression + ")";
        }
        reconstructed = ((TFunction.F_CACHE.REGISTER[f_id]==",")?"[":"")+reconstructed;
        for (int i = 0; i < Srcs.size(); ++i) {
            if (Srcs.get(i) != null) {
                if((TFunction.F_CACHE.REGISTER[f_id]==",")){
                    if(i==Srcs.size()-1){
                        reconstructed = reconstructed
                                + "]:(" + (
                                (Srcs.get(i) instanceof TFConstant)
                                        ?Srcs.get(i).toString().split("\\.")[0]
                                        :Srcs.get(i).toString()
                        ) +")";
                    }else{
                        reconstructed = reconstructed
                            + (
                            (Srcs.get(i) instanceof TFConstant)
                                   ?Srcs.get(i).toString().split("\\.")[0]
                                   :Srcs.get(i).toString()
                        );
                    }
                } else {
                    reconstructed = reconstructed + Srcs.get(i).toString();
                }
            } else {
                reconstructed = reconstructed + "(null)";
            }
            if (i < Srcs.size() - ((TFunction.F_CACHE.REGISTER[f_id]==",")?2:1)) {
                reconstructed = reconstructed + TFunction.F_CACHE.REGISTER[f_id];
            }
        }
        return "(" + reconstructed + ")";
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public abstract T activate(T[] input, int j);

    @Override
    public abstract T activate(T[] input);

    @Override
    public abstract T derive(T[] input, int index, int j);

    @Override
    public abstract T derive(T[] input, int index);

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public abstract double activate(final double[] input, int j);

    @Override
    public abstract double activate(final double[] input);

    @Override
    public abstract double derive(final double[] input, final int index, final int j);

    @Override
    public abstract double derive(final double[] input, final int index);

    //Activation stage 1: TFunction determination!
    //============================================================================================================================================================================================
    /**
     *  Responsible for handling functions with id's 0-9  (single input functions!)
     * */
    protected T tensorActivationOf(T input, boolean derive) {
        T output = T.factory.newTensor(input.shape(), input.translation());
        if(!derive && !this.isFlat){
            output.inject(new FBuilder().newBuild(f_id, 1, true).activate(new T[]{input}));
            output.add(input.find(TGraphLock.class));
            return output;
        }
        if(input.isOutsourced()){
            TDevice device = (TDevice) input.find(TDevice.class);
            device.calculate(new T[]{output, input}, f_id, (derive)?0:-1);
        }else{
            Calculation.foreach(input, output, (i, inputValue, outputValue)->{
                outputValue[i] = scalarActivationOf(inputValue[i], derive);
            });
        }
        if(!derive){
            TGraphBuilder.connect(output, new T[]{input}, this, true);
        }
        output.add(input.find(TGraphLock.class));
        return output;
    }

    /**
     *   Responsible for handling functions with multiple inputs!
     * */
    protected T tensorActivationOf(T[] input, int j, int d) {
        T output = T.factory.newTensor(input[0].shape(), input[0].translation());
        /**
         *   The code below deals with deep functions (non flat):
         * */
        if(d<0 && !isFlat){//only flat functions can be executed
            if(f_id<=9){
                output.inject(new FBuilder().newBuild(TFunction.F_CACHE.REGISTER[f_id]+"(I["+((j<0)?0:j)+"])", true).activate(input));
                return output;
            }else{
                if(TFunction.F_CACHE.REGISTER[f_id].length()!=1){
                    /**  SUMMATION, PI,
                     * */
                    T[] tsrs = activateSource(input);
                    output.inject(FBuilder.newBuild(TFunction.F_CACHE.REGISTER[f_id]+"(I[j])", true).activate(tsrs));
                    return output;
                }else if(f_id<=18){
                    /**      +, -, x, *, %, ....// TODO: move this to Function constructor!
                     * */
                    String operation = (TFunction.F_CACHE.REGISTER[f_id].length()>1)? TFunction.F_CACHE.REGISTER[f_id]:"";
                    T[] tsrs = activateSource(input, j, null);
                    for(int i=0; i<tsrs.length; i++){
                        operation += "I["+i+"]"+((i+1<tsrs.length)? TFunction.F_CACHE.REGISTER[f_id]:"");
                    }
                    if(j<0){
                        output.inject(FBuilder.newBuild(operation, true).activate(tsrs));
                    }else{
                        output.inject(FBuilder.newBuild(operation, true).activate(tsrs, j));
                    }
                    return output;
                }else{
                    /**
                     *    Tensor shape translation
                     * */
                    T[] tsrs = activateSource(input, j, new int[]{1});
                    if(j<0){
                        output.inject(new FBuilder().newBuild(f_id, tsrs.length, true).activate(tsrs));
                    }else{
                        output.inject(new FBuilder().newBuild(f_id, tsrs.length, true).activate(tsrs, j));
                    }
                    return output;
                }
            }
        }
        /**
         *  The following code is reached in flat functions only:
         * */
        TDevice device = (TDevice) input[0].find(TDevice.class);
        boolean onSameDevice = T.utility.shareGuestDevice(input);
        if(onSameDevice && TFunction.F_CACHE.REGISTER[f_id]!="," && !(TFunction.F_CACHE.REGISTER[f_id]=="x"&&d>-1)){
            if(device!=null){
                device.add(output);
            }
            for (int i = 0; i < input.length; i++) {
                device = (TDevice) input[i].find(TDevice.class);
                T[] tsrs = new T[1+input.length];
                tsrs[0]=output;
                for(int ii=1; ii<tsrs.length; ii++){
                    tsrs[ii]=input[ii-1];
                }
                device.calculate(tsrs, f_id, d);
            }
        }else{
            if(TFunction.F_CACHE.REGISTER[f_id]=="x"){
                if(d<0){
                    output = T.factory.convolution(input[0], input[1]);
                }else{
                    if(d==0){
                        output = input[1];
                    }else{
                        output = input[0];
                    }
                }
            }else if(TFunction.F_CACHE.REGISTER[f_id]==","){
                int[] newForm = new int[this.Srcs.size()-1];
                for(int i=0; i<this.Srcs.size()-1; i++ ){
                    TFunction fcn = this.Srcs.get(i);
                    if(fcn instanceof TFConstant){
                        newForm[i] = (int)((TFConstant)fcn).value();
                    }else{
                        T t = (j<0)
                            ?fcn.activate(input)
                            :fcn.activate(input, j);
                        newForm[i] = (int) T.factory.io.getFrom(t, 0);
                    }
                }
                T t = (j<0)
                    ?this.Srcs.get(Srcs.size()-1).activate(input)
                    :this.Srcs.get(Srcs.size()-1).activate(input, j);
                t = T.factory.reshaped(t, newForm, true);//t.reshape(newForm);

                if(d<0){
                    output.inject(t);
                }else{//reverse reshape:
                    /**
                     *      [3, 2, 4, 0, 1]
                     *      [0, 1, 2, 3, 4]
                     * */
                    int[] reversed = new int[newForm.length];
                    for(int i=0; i<newForm.length; i++){
                       // reversed[newForm[i]] = i;
                    }
                }
            } else {
                T[] tsrs = input;
                double[] inp = new double[tsrs.length];
                T finalOutput = output;
                output.foreach((i)->{
                    for (int ii = 0; ii < tsrs.length; ii++) {
                        inp[ii] = tsrs[ii].value()[i];
                    }
                    finalOutput.value()[i] = scalarActivationOf(inp, j, d);
                });
                //input = tsrs;
            }
        }
        if(d<0){
            T[] tsrs = input;
            TGraphBuilder.connect(output, tsrs, this, true);//TODO: remove f_id here and just pass on 'this'
        }
        return output;
    }

    private T[] activateSource(T[] input){
        T[] tsrs = new T[input.length];
        for(int i=0; i<tsrs.length; i++){
            tsrs[i] =  Srcs.get(0).activate(input, i);
        }
        return tsrs;
    }

    private T[] activateSource(T[] input, int j, int[] templateShape){
        T[] tsrs = new T[this.Srcs.size()];
        for(int i=0; i<tsrs.length; i++){//constants need to be figured out!
            if(Srcs.get(i) instanceof TFConstant){
                tsrs[i] = null;
            }else{
                tsrs[i] =
                    (j<0)
                        ?Srcs.get(i).activate(input)
                        :Srcs.get(i).activate(input, j);
                templateShape =
                    (templateShape==null)
                        ?tsrs[i].shape()
                        :templateShape;
            }
        }
        for(int i=0; i<tsrs.length; i++){
            tsrs[i] =
                (tsrs[i] != null)
                    ? tsrs[i]
                    : (j<0)
                        ?T.factory.newTensor(((TFConstant)this.Srcs.get(i)).value(), templateShape)
                        :T.factory.newTensor(Srcs.get(i).activate(new double[]{}, j), templateShape);
        }
        return tsrs;
    }

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    protected double scalarActivationOf(double input, boolean derive) {
        switch (f_id) {
            case 0: return Calculation.getReLuOf(       input, derive);
            case 1: return Calculation.getSigmoidOf(    input, derive);
            case 2: return Calculation.getTanhOf(       input, derive);
            case 3: return Calculation.getQuadraticOf(  input, derive);
            case 4: return Calculation.getLigmoidOf(    input, derive);
            case 5: return Calculation.getLinearOf(     input, derive);
            case 6: return Calculation.getGaussianOf(   input, derive);
            case 7: return Calculation.getAbsoluteOf(   input, derive);
            case 8: return Calculation.getSinusOf(      input, derive);
            case 9: return Calculation.getCosinusOf(    input, derive);
            default: return input;
        }
    }
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    protected double scalarActivationOf(double[] input, int j, int d) {
        switch (f_id) {
            case 10: return (j<0)? Calculation.getSummation(      input, d, Srcs) : Calculation.getSummation(      input, j, d, Srcs);
            case 11: return (j<0)? Calculation.getPI(             input, d, Srcs) : Calculation.getPI(             input, j, d, Srcs);
            case 12: return (j<0)? Calculation.getPowerOf(        input, d, Srcs) : Calculation.getPowerOf(        input, j, d, Srcs);
            case 13: return (j<0)? Calculation.getDivision(       input, d, Srcs) : Calculation.getDivision(       input, j, d, Srcs);
            case 14: return (j<0)? Calculation.getMultiplication( input, d, Srcs) : Calculation.getMultiplication( input, j, d, Srcs);
            case 15: return (j<0)? Calculation.getModulo(         input, d, Srcs) : Calculation.getModulo(         input, j, d, Srcs);
            case 16: return (j<0)? Calculation.getSubtraction(    input, d, Srcs) : Calculation.getSubtraction(    input, j, d, Srcs);
            case 17: return (j<0)? Calculation.getAddition(       input, d, Srcs) : Calculation.getAddition(       input, j, d, Srcs);
            case 18: return (j<0)? Calculation.getMultiplication( input, d, Srcs) : Calculation.getMultiplication( input, j, d, Srcs);
            default: return 0;
        }
    }

    protected static class Calculation {
        private interface Actor{ void apply(Integer i, double[] v1, double[] v2);}

        private static void foreach(T t1, T t2, Actor action){
            double[] inputValue = (t1.value()==null)?new double[t1.size()]:t1.value();
            double[] outputValue = (t2.value()==null)?new double[t2.size()]:t2.value();
            t1.foreach((i)->{ action.apply(i, inputValue, outputValue); });
            t2.setValue(outputValue);
            t1.setValue(inputValue);
        }
        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        //Activation stage 1: Activation f_id determination
        //============================================================================================================================================================================================
        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        public static double getReLuOf(double input, boolean derive) {
            double output;
            if (!derive) {
                if (input >= 0) {
                    output = (input + TFunction.F_CACHE.BIAS) * TFunction.F_CACHE.INCLINATION;
                } else {
                    output = (input + TFunction.F_CACHE.BIAS) * TFunction.F_CACHE.RELU_INCLINATION;
                }
                return output;
            } else {
                if (input >= 0) {
                    output = TFunction.F_CACHE.INCLINATION;
                } else {
                    output = TFunction.F_CACHE.RELU_INCLINATION;
                }
                return output;
            }
        }
        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        public static double getSigmoidOf(double input, boolean derive) {
            if (!derive) {
                return 1 / (1 + Math.pow(Math.E, (-(input + TFunction.F_CACHE.BIAS) * TFunction.F_CACHE.INCLINATION)));
            } else {
                return TFunction.F_CACHE.INCLINATION * (Math.pow(Math.E, -(input + TFunction.F_CACHE.BIAS) * TFunction.F_CACHE.INCLINATION)) / (Math.pow((1 + Math.pow(Math.E, -(input + TFunction.F_CACHE.BIAS) * TFunction.F_CACHE.INCLINATION)), 2) + 2 * Math.pow(Math.E, -(input + TFunction.F_CACHE.BIAS) * TFunction.F_CACHE.INCLINATION));
            }
        }
        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        public static double getTanhOf(double input, boolean derive) {
            if (!derive) {
                return ((input + TFunction.F_CACHE.BIAS) * TFunction.F_CACHE.INCLINATION) / Math.pow((1 + Math.pow(((input + TFunction.F_CACHE.BIAS) * TFunction.F_CACHE.INCLINATION), 2)), 0.5);
            } else {
                return (1 - Math.pow(((input + TFunction.F_CACHE.BIAS) / Math.pow((1 + Math.pow((input + TFunction.F_CACHE.BIAS), 2)), 0.5)), 2)) * TFunction.F_CACHE.INCLINATION;
            }
        }
        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        public static double getQuadraticOf(double input, boolean derive) {
            if (!derive) {
                return ((input + TFunction.F_CACHE.BIAS) * (input + TFunction.F_CACHE.BIAS) * TFunction.F_CACHE.INCLINATION);
            } else {
                return 2 * input * TFunction.F_CACHE.INCLINATION + 2 * TFunction.F_CACHE.BIAS * TFunction.F_CACHE.INCLINATION;
            }
        }
        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        public static double getLigmoidOf(double input, boolean derive) {
            if (!derive) {
                return (TFunction.F_CACHE.INCLINATION * (input + TFunction.F_CACHE.BIAS) + (Math.log(Math.pow(Math.E, -(input + TFunction.F_CACHE.BIAS) * TFunction.F_CACHE.INCLINATION) + 1) / Math.log(Math.E)));
            } else {
                return getSigmoidOf(input, false);
            }
        }
        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        public static double getLinearOf(double input, boolean derive) {
            if (!derive) {
                return TFunction.F_CACHE.INCLINATION * (input + TFunction.F_CACHE.BIAS);
            } else {
                return TFunction.F_CACHE.INCLINATION;
            }
        }
        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        public static double getGaussianOf(double input, boolean derive) {
            if (!derive) {
                return Math.pow(Math.E, -Math.pow(TFunction.F_CACHE.INCLINATION * (input + TFunction.F_CACHE.BIAS), 2));
            } else {
                return -2 * (TFunction.F_CACHE.INCLINATION * (input + TFunction.F_CACHE.BIAS)) * Math.pow(Math.E, -Math.pow(TFunction.F_CACHE.INCLINATION * (input + TFunction.F_CACHE.BIAS), 2));
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
        private static double getSummation(double[] input, int j, int d, ArrayList<TFunction> Variable){
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
        private static double getSummation(double[] input, int d, ArrayList<TFunction> Variable){
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
        private static double getPI(double[] input, int j, int d, ArrayList<TFunction> Variable){
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
        private static double getPI(double[] input, int d, ArrayList<TFunction> Variable){
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
        private static double getPowerOf(double[] input, int j, int d, ArrayList<TFunction> Variable){
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
        private static double getPowerOf(double[] input, int d, ArrayList<TFunction> Variable){
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
        private static double getDivision(double[] input, int j, int d, ArrayList<TFunction> Variable){
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
        private static double getDivision(double[] input, int d, ArrayList<TFunction> Variable){
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
        private static double getMultiplication(double[] input, int j, int d, ArrayList<TFunction> Variable){
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
        private static double getMultiplication(double[] input, int d, ArrayList<TFunction> Variable){
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
        private static double getModulo(double[] input, int j, int d, ArrayList<TFunction> Variable){
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
        private static double getModulo(double[] input, int d, ArrayList<TFunction> Variable){
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
        private static double getSubtraction(double[] input, int j, int d, ArrayList<TFunction> Variable){
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
        private static double getSubtraction(double[] input, int d, ArrayList<TFunction> Variable){
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
        private static double getAddition(double[] input, int j, int d, ArrayList<TFunction> Variable){
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
        private static double getAddition(double[] input, int d, ArrayList<TFunction> Variable){
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

}
