package neureka.main.core.modul.calc;

import neureka.main.core.base.data.AC;
import neureka.main.core.base.data.T;

import java.util.*;
import java.util.function.Supplier;

public class FunctionConstructor {

    private static final double shift = 0;
    private static final double inclination = 1;
    private static final double secondaryInclination = 0.01;
    private static final String[] register;
    private static final HashMap<String, Function> shared;
    static {
        register = new String[]{
                "relu", "sig", "tanh", "quad", "lig", "lin", "gaus", "abs", "sin", "cos",
                "sum", "prod",
                "^", "/", "*", "%", "-", "+", "x"
            };
        shared = new HashMap<>();
    }
    Function function = null;
    private ArrayList<Function> sources;
    /*
	     0:  ReLu;
		 1:  Sigmoid;
		 2:  Tanh;
		 3:  Quadratic;
		 4:  Ligmoid;
		 5:  Linear;
		 6:  Gaussian;
		 7:  abs;
		 8:  sin;
		 9:  cos;
		 10: sum;
		 11: prod;
		 12: ^;
		 13: /;
		 14: *;
		 15: %;
		 16: -;
		 17: +;
		 18: x; (conv)
	 */
    public FunctionConstructor() {
        this.sources = new ArrayList<Function>();
    }
    //=================================================================================================

    public Function newBuild(int f_id, int size, boolean tipReached){
        if(f_id==18&&tipReached){//<= occurs when
            size = 2;
        }
        if(f_id<10){
            return newBuild(register[f_id]+"(I[0])", tipReached);
        }else if(f_id<12){
            return newBuild(register[f_id]+"I[j]", tipReached);
        }else{
            String expression = "I[0]";
            for(int i=0; i<size-1; i++){
                expression+=register[f_id]+"I["+(i+1)+"]";
            }
            return newBuild(expression, tipReached);
        }
    }

    public Function newBuild(String expression){
        return newBuild(expression, false);
    }

    public Function newBuild(String expression, boolean tipReached){
        HashMap<String, Function> map = shared;
        expression = (expression.charAt(0)!='(')
            ?("("+expression)
            :(expression.charAt(expression.length()-1)!=')')
                ?expression+")"
                :expression;

        if(shared.containsKey(((tipReached)?"d":"")+expression)){
            return shared.get(((tipReached)?"d":"")+expression);
        }
        Function built = construct(expression, tipReached);
        if(built!=null){
            shared.put(((tipReached)?"d":"")+"("+built.toString()+")", built);
        }
        return built;
    }

    private Function construct(String expression, boolean tipReached)
    {
        if (expression == null) {
            expression = "";
        }
        if (expression == "") {
            Function newCore = new Function_Constant();
            newCore = newCore.newBuild("0");
            return newCore;
        }
        System.out.println("Equation packed: " + expression);
        expression = utility.unpackAndCorrect(expression);
        System.out.println("Equation unpacked: " + expression);
        List<String> Operations = new ArrayList<String>();
        List<String> Components = new ArrayList<String>();
        int i = 0;
        System.out.println("Expression length : " + expression.length());
        while (i < expression.length()) {
            System.out.println("\nPARSING [" + expression + "] at i:" + i + "\n================");
            System.out.println("parsing at " + i + " => searching for component!");
            final String newComponent = utility.parsedComponent(expression, i);
            if (newComponent != null) {
                System.out.println("Component found! : " + newComponent + " ");
                System.out.println(Components.size() + " < " + Operations.size() + " ?: true -> Components.e_add(" + newComponent + ")");
                if (Components.size() <= Operations.size()) {
                    Components.add(newComponent);
                }
                i += newComponent.length();
                System.out.println("parsing at " + i + " => searching for operation!");
                final String newOperation = utility.parsedOperation(expression, i);
                if (newOperation != null) {
                    System.out.println("Operation found! : " + newOperation + " ");
                    i += newOperation.length();
                    if (newOperation.length() <= 0) {
                        continue;
                    }
                    Operations.add(newOperation);
                }
                System.out.println("parsed operation: " + newOperation);
            } else {
                ++i;
            }
            System.out.println("Ending iteration with i:" + i + "\nExpression: " + expression + "\n=============================\n----------------------------");
            Components.forEach((c) -> { System.out.print("[" + c + "]"); });
            System.out.print(";\n");
            Operations.forEach((o) -> { System.out.print("[" + o + "]"); });
            System.out.print(";\n");
        }
        System.out.println("Components and OPerations parsed: " + Components);
        System.out.println("Operations and components parsed: " + Operations);
        //===
        int Count = FunctionConstructor.register.length;
        for (int j = FunctionConstructor.register.length; j > 0; --j) {
            if (!utility.containsOperation(FunctionConstructor.register[j - 1], Operations)) {
                --Count;
            } else {
                j = 0;
            }
        }
        int ID = 0;
        while (ID < Count) {
            final List<String> newOperations = new ArrayList<String>();
            final List<String> newComponents = new ArrayList<String>();
            System.out.println("current (" + register[ID] + ") and most low rank operation: " + register[Count - 1]);
            if (utility.containsOperation(FunctionConstructor.register[ID], Operations)) {
                String currentChain = null;
                boolean groupingOccured = false;
                boolean enoughtPresent = utility.numberOfOperationsWithin(Operations) > 1;// Otherwise: I[j]^4 goes nuts!
                if (enoughtPresent) {
                    String[] ComponentsArray = Components.toArray(new String[0]);
                    int length = ComponentsArray.length;
                    System.out.println("Iterating over " + length + " components: ");
                    for (int Ci = 0; Ci < length; Ci++)
                    {
                        System.out.println("\nIteration " + Ci + "/" + length);
                        System.out.println("====================");
                        String currentComponent = null;
                        currentComponent = ComponentsArray[Ci];//ComponentIterator.next();
                        String currentOperation = null;
                        if (Operations.size() > Ci) {//OperationIterator.hasNext()
                            currentOperation = Operations.get(Ci);//OperationIterator.next();
                        }
                        System.out.println("Component: " + currentComponent);
                        System.out.println("Operation: " + currentOperation);
                        System.out.println("Chain: " + currentChain);
                        if (currentOperation != null) {
                            if (currentOperation.equals(FunctionConstructor.register[ID])) {
                                final String newChain =
                                        utility.groupBy(FunctionConstructor.register[ID], currentChain, currentComponent, currentOperation);
                                System.out.println("newChain: " + newChain);
                                if (newChain != null) {
                                    currentChain = newChain;
                                }
                                System.out.println("This needed to be grouped");
                                groupingOccured = true;
                            } else {
                                if (currentChain == null) {
                                    newComponents.add(currentComponent);
                                } else if (currentChain != null) {
                                    newComponents.add(currentChain + currentComponent); //= String.valueOf(currentChain) + currentComponent
                                }
                                newOperations.add(currentOperation);
                                groupingOccured = true;
                                currentChain = null;
                            }
                        } else {
                            if (currentChain == null) {
                                newComponents.add(currentComponent);
                            } else {
                                newComponents.add(currentChain + currentComponent);
                                groupingOccured = true;
                            }
                            currentChain = null;
                        }
                        System.out.println("----------------\nComponent: " + currentComponent);
                        System.out.println("Operation: " + currentOperation);
                        System.out.println("Chain: " + currentChain + "\n");
                    }
                }
                if (groupingOccured) {
                    Operations = newOperations;
                    Components = newComponents;
                    System.out.println("Grouping occured:");
                } else {
                    System.out.println("Grouping did not occure:");
                }
            }
            System.out.println("ID:" + ID + " => register[ID]: " + FunctionConstructor.register[ID]);
            System.out.println("Operations: " + Operations);
            System.out.println("Components: " + Components);
            ++ID;
        }//closed while(ID < Count)
        System.out.println("==========================================================================================================");
        //---------------------------------------------------------
        // identifying function id:
        int f_id = 0;
        if (Operations.size() >= 1) {
            System.out.println("Operation 0 : " + Operations.get(0));
            for (int k = 0; k < FunctionConstructor.register.length; ++k) {
                if (FunctionConstructor.register[k].equals(Operations.get(0))) {
                    f_id = k;
                }
            }
        }
        // building sources and function:
        if (Components.size() == 1) {
            System.out.println("Only one component left -> no operations! -> testing for function:");
            System.out.println("parsing at " + 0 + " ...(possibly a function!)");
            String possibleFunction = utility.parsedOperation(Components.get(0).toLowerCase(), 0);
            System.out.println("Function ?: " + possibleFunction);
            if (possibleFunction != null) {
                if (possibleFunction.length() > 1) {
                    for (int Oi = 0; Oi < register.length; Oi++) {
                        if (register[Oi].equals(possibleFunction)) {
                            f_id = Oi;
                            Function newCore = new FunctionConstructor()
                                    .newBuild(
                                            utility.parsedComponent(Components.get(0), possibleFunction.length())
                                    );
                            sources.add(newCore);
                            function = construction.createFunction(f_id, sources, tipReached);
                            return this.function;
                        }
                    }
                }
            }
            //function = construction.createFunction(f_id, sources);
            System.out.println("Function: " + possibleFunction);
            //---
            System.out.println("1 comonent left: -> unpackAndCorrect(component)");
            String component = utility.unpackAndCorrect(Components.get(0));
            System.out.println("component: " + component);
            System.out.println("Checking if component is variable (value/input): ");
            if ((component.charAt(0) <= '9' && component.charAt(0) >= '0') || component.charAt(0) == '-' || component.charAt(0) == '+') {
                Function newFunction = new Function_Constant();
                newFunction = newFunction.newBuild(component);
                System.out.println("is value leave! -> return newValueLeave.newBuilt(component)");
                return newFunction;
            }
            if (component.charAt(0) == 'i' || component.charAt(0) == 'I') {
                Function newFunction = new Function_Input();
                newFunction = newFunction.newBuild(component);
                System.out.println("value leave! -> return newInputLeave.newBuilt(component)");
                return newFunction;
            }
            System.out.println("Component is not of type Leave! -> component = utility.cleanedHeadAndTail(component); ");
            //If the component did not trigger variable creation: =>Cleaning!
            component = utility.cleanedHeadAndTail(component);

            Function newBuild;
            System.out.println("new build: Function newBuild = (Function)new FunctionConstructor();");
            System.out.println("newBuild = newBuild.newBuild(component);");
            newBuild = new FunctionConstructor().newBuild(component);
            System.out.println("-> return newBuild;");
            return newBuild;
        } else {// More than one component left:
            if(register[f_id]=="x"){
                Components = rebindPairwise(Components, f_id);
            }
            final ListIterator<String> ComponentIterator2 = Components.listIterator();
            while (ComponentIterator2.hasNext()) {
                final String currentComponent2 = ComponentIterator2.next();
                System.out.println("this.Input.add(newCore2.newBuild("+currentComponent2+")); Input.size(): "+this.sources.size());
                //Function newCore2 = new FunctionConstructor();
                Function newCore2 = new FunctionConstructor().newBuild(currentComponent2);//Dangerous recursion lives here!
                this.sources.add(newCore2);
                if (newCore2 != null) {
                    System.out.println("newCore2 != null");
                }
            }
            this.sources.trimToSize();
            function = construction.createFunction(f_id, sources, tipReached);
            if (this.sources.size() == 1) {
                return this.sources.get(0);
            }
            if (this.sources.size() == 0) {
                return null;
            }
            ArrayList<Function> newVariable = new ArrayList<Function>();
            for (int Vi = 0; Vi < sources.size(); Vi++) {
                if (sources.get(Vi) != null) {
                    newVariable.add(sources.get(Vi));
                }
            }
            sources = newVariable;
            return this.function;
        }
    }
    private List<String> rebindPairwise(List<String> components, int f_id){
        if(components.size()>2){
            String newComponent = "("+components.get(0)+register[f_id]+components.get(1)+")";
            components.remove(components.get(0));
            components.remove(components.get(0));
            components.add(0, newComponent);
            components = rebindPairwise(components, f_id);
        }
        return components;
    }
    //Interpretation functions end.
    //======================================================================================
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public String toString() {
        return "(" + function.toString() + ")";
    }
    ////~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    //Activation stage 1: Function determination!
    //============================================================================================================================================================================================
    /*
    case 0:  return "relu";
	case 1:  return "sig";
	case 2:  return "tanh";
	case 3:  return "quad";
	case 4:  return "lig";
	case 5:  return "lin";
	case 6:  return "gaus";
	case 7:  return "abs";
	case 8:  return "sin";
	case 9:  return "cos";

	case 10: return "sum";
	case 11: return "prod";

	case 12: return "^";
	case 13: return "/";
	case 14: return "*";
	case 15: return "%";
	case 16: return "-";
	case 17: return "+";

    case 18: return "conv";
    * */
    private static class construction{
        public static Function createFunction(int f_id, ArrayList<Function> Srcs, boolean tipReached){
            boolean[] isFlat = {false};//=> !isFlat
            Srcs.forEach((v)->{
                isFlat[0] = (
                    (!(v instanceof Function_Input)) &&
                    (!(v instanceof Function_Variable)) &&
                    (!(v instanceof Function_Constant))
                )
                || isFlat[0];
            });
            isFlat[0] = !isFlat[0];

            Supplier<String> string = () -> {
                String reconstructed = "";
                if (Srcs.size() == 1 && register[f_id].length() > 1) {
                    String expression = Srcs.get(0).toString();
                    if (expression.charAt(0) == '(' && expression.charAt(expression.length() - 1) == ')') {
                        return register[f_id] + expression;
                    }
                    return register[f_id] + "(" + expression + ")";
                }
                for (int i = 0; i < Srcs.size(); ++i) {
                    if (Srcs.get(i) != null) {
                        reconstructed = reconstructed + Srcs.get(i).toString();
                    } else {
                        reconstructed = reconstructed + "(null)";
                    }
                    if (i != Srcs.size() - 1) {
                        reconstructed = reconstructed + register[f_id];
                    }
                }
                return "(" + reconstructed + ")";
            };

            if(f_id<9) {// FUNCTIONS:
                return new Function(){
                    @Override
                    public boolean isFlat(){
                        return  isFlat[0];
                    }

                    @Override
                    public int id() {
                        return f_id;
                    }

                    @Override
                    public Function newBuild(String expression) {
                        return new FunctionConstructor().newBuild(expression);
                    }
                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public String toString() {
                        return string.get();//register[f_id];
                    }
                    @Override
                    public T activate(T[] input, int j) {
                        return calculation.tensorActivationOf(Srcs.get(0).activate(input, j), f_id, false, tipReached, isFlat[0]);
                    }
                    @Override
                    public T activate(T[] input) {
                        return calculation.tensorActivationOf(Srcs.get(0).activate(input), f_id, false, tipReached, isFlat[0]);
                    }
                    @Override
                    public T derive(T[] input, int d, int j) {
                        return calculation.tensorActivationOf(Srcs.get(0).activate(input, j), f_id, true, tipReached, isFlat[0]);
                    }
                    @Override
                    public T derive(T[] input, int d) {
                        return calculation.tensorActivationOf(Srcs.get(0).activate(input), f_id, true, tipReached, isFlat[0]);
                    }
                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input, int j) {
                        return calculation.scalarActivationOf(Srcs.get(0).activate(input, j), f_id, false);
                    }
                    @Override
                    public double activate(final double[] input) {
                        return calculation.scalarActivationOf(Srcs.get(0).activate(input), f_id, false);
                    }
                    @Override
                    public double derive(final double[] input, final int index, final int j) {
                        return calculation.scalarActivationOf(Srcs.get(0).activate(input, j), f_id, true)
                                * Srcs.get(0).derive(input, index, j);
                    }
                    @Override
                    public double derive(final double[] input, final int index) {
                        return calculation.scalarActivationOf(Srcs.get(0).activate(input), f_id, true)
                                * Srcs.get(0).derive(input, index);
                    }
                };
            }else{
                    return new Function(){
                        @Override
                        public boolean isFlat(){
                            return  isFlat[0];
                        }

                        @Override
                        public int id() {
                            return f_id;
                        }

                        @Override
                        public Function newBuild(String expression){
                            return new FunctionConstructor().newBuild(expression);
                        }
                        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        @Override
                        public String toString() {
                            return string.get();//register[f_id];// flat&rec -> flat&!rec
                        }
                        @Override
                        public T activate(T[] input, int j) {
                            return calculation.tensorActivationOf(input, f_id, j, -1, Srcs, tipReached, isFlat[0]);
                        }
                        @Override
                        public T activate(T[] input) {
                            return calculation.tensorActivationOf(input, f_id, -1, -1, Srcs, tipReached, isFlat[0]);
                        }
                        @Override
                        public T derive(T[] input, int d, int j) {
                            return calculation.tensorActivationOf(input, f_id, j, d, Srcs, tipReached, isFlat[0]);
                        }
                        @Override
                        public T derive(T[] input, int d) {
                            return calculation.tensorActivationOf(input, f_id, -1, d, Srcs, tipReached, isFlat[0]);
                        }
                        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        @Override
                        public double activate(final double[] input, int j) {
                            return calculation.scalarActivationOf(input, f_id, j, -1, Srcs);
                        }
                        @Override
                        public double activate(final double[] input) {
                            return calculation.scalarActivationOf(input, f_id, -1, -1, Srcs);
                        }
                        @Override
                        public double derive(final double[] input, final int d, final int j) {
                            return calculation.scalarActivationOf(input, f_id, j, d, Srcs);
                        }
                        @Override
                        public double derive(final double[] input, final int d) {
                            return calculation.scalarActivationOf(input, f_id, -1, d, Srcs);
                        }
                    };

            }
        }
    }
    
    private static class calculation
    {
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
                output.addModule(new AC(output, new T[]{input}, f_id, true, false));
                return output;
            }
            if(input.isOutsourced()){
                TDevice device = (TDevice) input.find(TDevice.class);
                device.calculate(new T[]{output, input}, f_id, (derive)?0:-1);
            }else{
                foreach(input, output, (i, inputValue, outputValue)->{
                    outputValue[i] = calculation.scalarActivationOf(inputValue[i], f_id, derive);
                });
            }
            if(!derive){//&& tipReached
                output.addModule(new AC(output, new T[]{input}, f_id, false, (isFlat)?(!tipReached)?true:false:false));
            }
            return output;
        }
        //TODO: j is not handled properly!!!
        /**
         *   Responsible for handling functions with multiple inputs!
         * */
        public static T tensorActivationOf(T[] input, int f_id, int j, int d, ArrayList<Function> Srcs, boolean tipReached, boolean isFlat) {
            T output = T.factory.newTensor(input[0].shape(), input[0].translation());
            if(d<0 && !isFlat){//implies !tipReached ==true // only flat functions can be executed
                if(f_id<9){
                    output.addModule(new AC(output, input, register[f_id]+"(I["+((j<0)?0:j)+"])", true, false));
                    return output;
                }else{
                    if(register[f_id].length()!=1){
                        /**  SUMMATION, PI,
                         * */
                        T[] tsrs = new T[input.length];
                            for(int i=0; i<tsrs.length; i++){
                                tsrs[i] =  Srcs.get(0).activate(input, i);
                            }// THIS NEEDS TO BE SOLVED!!
                            output.addModule(new AC(output, tsrs, register[f_id]+"(I[j])", true, false));
                            return output;
                    }else{
                        /**      +, -, x, *, %, ....
                         * */
                        String operation = (register[f_id].length()>1)?register[f_id]:"";
                        T[] tsrs = new T[Srcs.size()];
                        boolean constantFound = false;
                        T template = null;
                        for(int i=0; i<tsrs.length; i++){//constants need to be figured out!
                            if(Srcs.get(i) instanceof Function_Constant){
                                tsrs[i] = null;
                                constantFound = true;
                            }else{
                                tsrs[i] = (j<0)?Srcs.get(i).activate(input):Srcs.get(i).activate(input, j);
                                template = tsrs[i];
                            }
                            operation += "I["+i+"]"+((i+1<tsrs.length)?register[f_id]:"");
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
                        output.addModule(new AC(output, tsrs, operation, true, false));
                        return output;
                    }
                }
            }
            /**
             *  The following code is reached in flat functions only:
             * */
            TDevice device = (TDevice) input[0].find(TDevice.class);
            boolean onSameDevice = T.utility.shareGuestDevice(input);
            if(onSameDevice){
                if(device!=null){
                    device.add(output);
                }
                for (int ti = 0; ti < input.length; ti++) {
                    device = (TDevice) input[ti].find(TDevice.class);
                    T[] tsrs = new T[1+input.length];
                    tsrs[0]=output;
                    for(int tii=1; tii<tsrs.length; tii++){
                        tsrs[tii]=input[tii-1];
                    }
                    device.calculate(tsrs, f_id, d);
                }
            }else{
                if(register[f_id]=="x"){
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
                        finalOutput.value()[i] = calculation.scalarActivationOf(inp, f_id, j, d, Srcs);
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
                output.addModule(new AC(output, tsrs, f_id, false, (isFlat)?(!tipReached)?true:false:false));
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
                    output = (input + shift) * inclination;
                } else {
                    output = (input + shift) * secondaryInclination;
                }
                return output;
            } else {
                if (input >= 0) {
                    output = inclination;
                } else {
                    output = secondaryInclination;
                }
                return output;
            }
        }
        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        public static double getSigmoidOf(double input, boolean derive) {
            if (!derive) {
                return 1 / (1 + Math.pow(Math.E, (-(input + shift) * inclination)));
            } else {
                return inclination * (Math.pow(Math.E, -(input + shift) * inclination)) / (Math.pow((1 + Math.pow(Math.E, -(input + shift) * inclination)), 2) + 2 * Math.pow(Math.E, -(input + shift) * inclination));
            }
        }
        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        public static double getTanhOf(double input, boolean derive) {
            if (!derive) {
                return ((input + shift) * inclination) / Math.pow((1 + Math.pow(((input + shift) * inclination), 2)), 0.5);
            } else {
                return (1 - Math.pow(((input + shift) / Math.pow((1 + Math.pow((input + shift), 2)), 0.5)), 2)) * inclination;
            }
        }
        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        public static double getQuadraticOf(double input, boolean derive) {
            if (!derive) {
                return ((input + shift) * (input + shift) * inclination);
            } else {
                return 2 * input * inclination + 2 * shift * inclination;
            }
        }
        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        public static double getLigmoidOf(double input, boolean derive) {
            if (!derive) {
                return (inclination * (input + shift) + (Math.log(Math.pow(Math.E, -(input + shift) * inclination) + 1) / Math.log(Math.E)));
            } else {
                return getSigmoidOf(input, false);
            }
        }
        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        public static double getLinearOf(double input, boolean derive) {
            if (!derive) {
                return inclination * (input + shift);
            } else {
                return inclination;
            }
        }
        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        public static double getGaussianOf(double input, boolean derive) {
            if (!derive) {
                return Math.pow(Math.E, -Math.pow(inclination * (input + shift), 2));
            } else {
                return -2 * (inclination * (input + shift)) * Math.pow(Math.E, -Math.pow(inclination * (input + shift), 2));
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


    /**
     *  UTILITY
     *  for parsing:
     *
     * */
    private static class utility {

        public static int numberOfOperationsWithin(final List<String> operations) {
            int Count = 0;
            for (int i = 0; i < FunctionConstructor.register.length; ++i) {
                if (utility.containsOperation(FunctionConstructor.register[i], operations)) {
                    ++Count;
                }
            }
            return Count;
        }

        public static String parsedOperation(final String equation, final int i) {
            String operation = "";
            if (equation.length() <= i) {
                return null;
            }
            operation = "";
            for (int Si = i; Si < equation.length(); ++Si) {
                operation = (operation) + equation.charAt(Si);
                if (utility.isBasicOperation(operation)) {
                    return operation;
                }
            }
            return null;
        }

        public static String parsedComponent(final String equation, final int i) {
            if (equation.length() <= i) {
                return null;
            }
            String component = "";
            int bracketDepth = 0;
            component = "";
            System.out.print("Start char: " + equation.charAt(i) + "\n");
            for (int Ei = i; Ei < equation.length(); ++Ei) {
                if (equation.charAt(Ei) == ')') {
                    --bracketDepth;
                } else if (equation.charAt(Ei) == '(') {
                    ++bracketDepth;
                }
                System.out.print("d[" + bracketDepth + "]:[  " + equation.charAt(Ei) + "  ], ");
                if (bracketDepth == 0) {
                    String possibleOperation = "";
                    for (int Sii = Ei + 1; Sii < equation.length(); ++Sii) {
                        possibleOperation = String.valueOf(possibleOperation) + equation.charAt(Sii);
                        if (utility.isBasicOperation(possibleOperation)) {
                            component = String.valueOf(component) + equation.charAt(Ei);
                            System.out.print("\n");
                            return component;
                        }
                    }
                }
                component = String.valueOf(component) + equation.charAt(Ei);
            }
            System.out.print("\n");
            return component;
        }

        public static boolean containsOperation(final String operation, final List<String> operations) {
            final ListIterator<String> OperationIterator = operations.listIterator();
            while (OperationIterator.hasNext()) {
                final String currentOperation = OperationIterator.next();
                if (currentOperation.equals(operation)) {
                    return true;
                }
            }
            return false;
        }

        public static boolean isBasicOperation(final String operation) {
            if (operation.length() > 8) {
                return false;
            }
            for (int i = 0; i < FunctionConstructor.register.length; ++i) {
                System.out.print(register[i] + " =?= " + operation + " -:|:- ");
                if (FunctionConstructor.register[i].equals(operation)) {
                    System.out.println("");
                    return true;
                }
            }
            return false;
        }
        public static String groupBy(final String operation, final String currentChain, final String currentComponent, final String currentOperation) {
            String group = null;
            if (currentOperation != null) {
                if (currentOperation.equals(operation)) {
                    group = String.valueOf(currentComponent) + currentOperation;
                    if (currentChain != null) {
                        group = String.valueOf(currentChain) + group;
                    }
                }
            } else if (currentChain != null) {
                group = String.valueOf(currentChain) + currentComponent;
            }
            return group;
        }
        private static boolean isWeired(char c) {
            if (c == '"') {
                return true;
            }
            if ('§' == c) {
                return true;
            }
            if (c == '$') {
                return true;
            }
            if (c == '%') {
                return true;
            }
            if (c == '&') {
                return true;
            }
            if (c == '=') {
                return true;
            }
            if (c == '#') {
                return true;
            }
            if (c == '|') {
                return true;
            }
            if (c == '~') {
                return true;
            }
            if (c == ':') {
                return true;
            }
            if (c == ';') {
                return true;
            }
            if (c == '@') {
                return true;
            }
            if (c == 'Ü') {
                return true;
            }
            if (c == 'Ä') {
                return true;
            }
            if (c == 'Ö') {
                return true;
            }
            if (c == '?') {
                return true;
            }
            if (c == '´') {
                return true;
            }
            if (c == '\\') {
                return true;
            }
            if (c == '>') {
                return true;
            }
            if (c == '<') {
                return true;
            }
            if (c == ' ') {
                return true;
            }
            return false;
        }
        public static String removeHeadAndTail(final String equation) {
            String corrected = "";
            for (int i = 1; i < equation.length() - 1; ++i) {
                corrected = String.valueOf(corrected) + equation.charAt(i);
            }
            return corrected;
        }
        public static String cleanedHeadAndTail(String exp) {
            System.out.println("Unclean component: " + exp);
            int Ci = 0;
            String Updated = "";
            boolean condition = true;
            while (condition) {
                if (utility.isWeired(exp.charAt(Ci)) || (exp.charAt(Ci) >= 'A' && exp.charAt(Ci) <= 'Z') || (exp.charAt(Ci) >= 'a' && exp.charAt(Ci) <= 'z')) {
                    System.out.print("C: " + exp.charAt(Ci) + "; ");
                    Ci++;
                } else {
                    condition = false;
                }
                if (Ci == exp.length()) {
                    condition = false;
                }
            }
            for (int Gi = Ci; Gi < exp.length(); Gi++) {
                Updated += exp.charAt(Gi);
            }
            exp = Updated;
            Updated = "";
            System.out.print("\nUpdated: " + exp + "  \n");
            if (exp.length() > 0) {
                Ci = 0;
                condition = true;
                int l = exp.length() - 1;
                while (condition) {
                    if (utility.isWeired(exp.charAt(Ci)) || (exp.charAt(l - Ci) >= 'A' && exp.charAt(l - Ci) <= 'Z') || (exp.charAt(l - Ci) >= 'a' && exp.charAt(l - Ci) <= 'z')) {
                        System.out.print("C: " + exp.charAt(l - Ci) + "; ");
                        Ci++;
                    } else {
                        condition = false;
                    }
                    if (l - Ci < 0) {
                        condition = false;
                    }
                }
                for (int Gi = 0; Gi <= l - Ci; Gi++) {
                    Updated += exp.charAt(Gi);
                }
                exp = Updated;
            }
            if (exp.length() > 0) {
                if (exp.charAt(0) == '(' && exp.charAt(exp.length() - 1) != ')') {
                    exp = utility.removeHeadAndTail(exp);
                }
                if (exp.charAt(exp.length() - 1) == ')' && exp.charAt(0) != '(') {
                    exp = utility.removeHeadAndTail(exp);
                }
            }
            System.out.println("Cleaned component: " + exp);
            return exp;
        }
        public static String unpackAndCorrect(String equation) {
            if (equation == null) {
                return null;
            }
            if (equation.length() == 0) {
                return "";
            }
            if (equation.equals("()")) {
                return "";
            }

            equation = equation.replace("lig", register[4]);
            equation = equation.replace("ligmoid", register[4]);
            equation = equation.replace("softplus", register[4]);
            equation = equation.replace("spls", register[4]);
            equation = equation.replace("ligm", register[4]);
            equation = equation.replace("linear", register[5]);
            equation = equation.replace("sigmoid", register[1]);
            equation = equation.replace("quadratic", register[3]);
            equation = equation.replace("quadr", register[3]);
            equation = equation.replace("gaussian", register[6]);
            equation = equation.replace("gauss", register[6]);
            equation = equation.replace("summation", register[7]);
            equation = equation.replace("product", register[8]);
            equation = equation.replace("absolute", register[9]);
            /*
             * OCode = new String[]
             * {"relu", "sig", "tanh", "quad", "lig", "lin", "gaus",
             *  "sum",  "prod",
             *  "^",     "/",  "*",     "%",   "+",   "-" };
             * */

            int bracketDepth = 0;
            for (int Ei = 0; Ei < equation.length(); ++Ei) {
                if (equation.charAt(Ei) == ')') {
                    --bracketDepth;
                } else if (equation.charAt(Ei) == '(') {
                    ++bracketDepth;
                }
            }
            if (bracketDepth != 0) {
                if (bracketDepth < 0) {
                    for (int Bi = 0; Bi < -bracketDepth; ++Bi) {
                        equation = "(" + equation;
                    }
                } else if (bracketDepth > 0) {
                    for (int Bi = 0; Bi < bracketDepth; ++Bi) {
                        equation = String.valueOf(equation) + ")";
                    }
                }
            }

            boolean parsing = true;
            boolean needsStitching = false;
            while (parsing && equation.charAt(0) == '(' && equation.charAt(equation.length() - 1) == ')') {
                bracketDepth = 0;
                needsStitching = true;
                for (int i = 0; i < equation.length(); ++i) {
                    if (equation.charAt(i) == ')') {
                        --bracketDepth;
                    } else if (equation.charAt(i) == '(') {
                        ++bracketDepth;
                    }
                    if (bracketDepth == 0 && i != equation.length() - 1) {
                        needsStitching = false;
                    }
                }
                if (needsStitching) {
                    equation = utility.removeHeadAndTail(equation);
                } else {
                    parsing = false;
                }
            }
            return equation;
        }


    }


}
