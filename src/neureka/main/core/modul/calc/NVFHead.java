package neureka.main.core.modul.calc;

import java.util.*;

public class NVFHead implements NVFunction {
    private interface NFOperation {
        public String expression();

        public double activate(double[] input);

        public double activate(double[] input, int j);

        public double derive(double[] input, int index);

        public double derive(double[] input, int index, int j);
    }

    private static final double shift = 0;
    private static final double inclination = 1;
    private static final double secondaryInclination = 0.01;
    private static final String[] OperationRegister;

    static {
        OperationRegister = new String[]
                {"relu", "sig", "tanh", "quad", "lig", "lin", "gaus", "sum", "prod", "abs", "sin", "cos",
                        "^", "/", "*", "%", "-", "+"};
    }

    NFOperation Opertaion = null;
    private ArrayList<NVFunction> Variable;

    /*
	     case 0:  ReLuActivation;
		 case 1:  SigmoidActivation;
		 case 2:  TanhActivation;
		 case 3:  QuadraticActivation;
		 case 4:  LigmoidActivation;
		 case 5:  LinearActivation;
		 case 6:  GaussianActivation;
		 case 7:  sum;
		 case 8:  prod;
		 case 9:  abs;
		 case 10: sin;
		 case 11: cos;
		 case 12: ^;
		 case 13: /;
		 case 14: *;
		 case 15: %;
		 case 16: -;
		 case 17: +;
	 */
    public NVFHead() {
        this.Variable = new ArrayList<NVFunction>();
    }

    //=================================================================================================
    @Override
    public NVFunction newBuild(String expression) {
        if (expression == null) {
            expression = "";
        }
        if (expression == "") {
            NVFunction newCore = new NVFValueLeave();
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
            Components.forEach((c) -> {
                System.out.print("[" + c + "]");
            });
            System.out.print(";\n");
            Operations.forEach((o) -> {
                System.out.print("[" + o + "]");
            });
            System.out.print(";\n");
        }
        System.out.println("Components and OPerations parsed: " + Components);
        System.out.println("Operations and components parsed: " + Operations);
        //===
        int Count = NVFHead.OperationRegister.length;
        for (int j = NVFHead.OperationRegister.length; j > 0; --j) {
            if (!utility.containsOperation(NVFHead.OperationRegister[j - 1], Operations)) {
                --Count;
            } else {
                j = 0;
            }
        }
        int ID = 0;
        while (ID < Count) {
            final List<String> newOperations = new ArrayList<String>();
            final List<String> newComponents = new ArrayList<String>();
            System.out.println("current (" + OperationRegister[ID] + ") and most low rank operation: " + OperationRegister[Count - 1]);
            if (utility.containsOperation(NVFHead.OperationRegister[ID], Operations)) {
                String currentChain = null;
                boolean groupingOccured = false;
                boolean enoughtPresent = utility.amountOfOperationsWithin(Operations) > 1;// Otherwise: I[j]^4 goes nuts!
                if (enoughtPresent) {
                    String[] ComponentsArray = Components.toArray(new String[0]);
                    int length = ComponentsArray.length;
                    System.out.println("Iterating over " + length + " components: ");
                    for (int Ci = 0; Ci < length; Ci++)
                    //while (ComponentIterator.hasNext())
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
                            if (currentOperation.equals(NVFHead.OperationRegister[ID])) {
                                final String newChain =
                                        utility.groupBy(NVFHead.OperationRegister[ID], currentChain, currentComponent, currentOperation);
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
            System.out.println("ID:" + ID + " => OperationRegister[ID]: " + NVFHead.OperationRegister[ID]);
            System.out.println("Operations: " + Operations);
            System.out.println("Components: " + Components);
            ++ID;
        }//closed while(ID < Count)
        System.out.println("==========================================================================================================");
        //Operation setup:
        NFOperation[] OperationCollection = {
                //0: ReLu
                new NFOperation() {
                    @Override
                    public String expression() {
                        return "relu";
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input, int j) {
                        return ActivationOf(Variable.get(0).activate(input, j), 0, false);
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input) {
                        return ActivationOf(Variable.get(0).activate(input), 0, false);
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index, final int j) {
                        return ActivationOf(Variable.get(0).activate(input, j), 0, true) * Variable.get(0).derive(input, index, j);
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index) {
                        return ActivationOf(Variable.get(0).activate(input), 0, true) * Variable.get(0).derive(input, index);
                    }
                }
                ,
                //1: Sigmoid:
                new NFOperation() {
                    @Override
                    public String expression() {
                        return "sig";
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input, int j) {
                        return ActivationOf(Variable.get(0).activate(input, j), 1, false);
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input) {
                        return ActivationOf(Variable.get(0).activate(input), 1, false);
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index, final int j) {
                        return ActivationOf(Variable.get(0).activate(input, j), 1, true) * Variable.get(0).derive(input, index, j);
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index) {
                        return ActivationOf(Variable.get(0).activate(input), 1, true) * Variable.get(0).derive(input, index);
                    }
                }
                ,
                //2: Tanh
                new NFOperation() {
                    @Override
                    public String expression() {
                        return "tanh";
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input, int j) {
                        return ActivationOf(Variable.get(0).activate(input, j), 2, false);
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input) {
                        return ActivationOf(Variable.get(0).activate(input), 2, false);
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index, final int j) {
                        return ActivationOf(Variable.get(0).activate(input, j), 2, true) * Variable.get(0).derive(input, index, j);
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index) {
                        return ActivationOf(Variable.get(0).activate(input), 2, true) * Variable.get(0).derive(input, index);
                    }
                }
                ,
                //3: Quadratic:
                new NFOperation() {
                    @Override
                    public String expression() {
                        return "quad";
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input, int j) {
                        return ActivationOf(Variable.get(0).activate(input, j), 3, false);
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input) {
                        return ActivationOf(Variable.get(0).activate(input), 3, false);
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index, final int j) {
                        return ActivationOf(Variable.get(0).activate(input, j), 3, true) * Variable.get(0).derive(input, index, j);
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index) {
                        return ActivationOf(Variable.get(0).activate(input), 3, true) * Variable.get(0).derive(input, index);
                    }
                }
                ,
                //4: Ligmoid:
                new NFOperation() {
                    @Override
                    public String expression() {
                        return "lig";
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input, int j) {
                        return ActivationOf(Variable.get(0).activate(input, j), 4, false);
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input) {
                        return ActivationOf(Variable.get(0).activate(input), 4, false);
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index, final int j) {
                        return ActivationOf(Variable.get(0).activate(input, j), 4, true) * Variable.get(0).derive(input, index, j);
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index) {
                        return ActivationOf(Variable.get(0).activate(input), 4, true) * Variable.get(0).derive(input, index);
                    }
                }
                ,
                //5: Linear:
                new NFOperation() {
                    @Override
                    public String expression() {
                        return "lin";
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input, int j) {
                        return ActivationOf(Variable.get(0).activate(input, j), 5, false);
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input) {
                        return ActivationOf(Variable.get(0).activate(input), 5, false);
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index, final int j) {
                        return ActivationOf(Variable.get(0).activate(input, j), 5, true) * Variable.get(0).derive(input, index, j);
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index) {
                        return ActivationOf(Variable.get(0).activate(input), 5, true) * Variable.get(0).derive(input, index);
                    }
                }
                ,
                //6: Gaussian
                new NFOperation() {
                    @Override
                    public String expression() {
                        return "gaus";
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input, int j) {
                        return ActivationOf(Variable.get(0).activate(input, j), 6, false);
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input) {
                        return ActivationOf(Variable.get(0).activate(input), 6, false);
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index, final int j) {
                        return ActivationOf(Variable.get(0).activate(input, j), 6, true) * Variable.get(0).derive(input, index, j);
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index) {
                        return ActivationOf(Variable.get(0).activate(input), 6, true) * Variable.get(0).derive(input, index);
                    }
                }
                ,
                //7: sum:
                new NFOperation() {
                    @Override
                    public String expression() {
                        return "sum";
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input, int j) {
                        double sum = 0;
                        boolean nothingDone = true;
                        for (int Ii = 0; Ii < input.length; Ii++) {
                            if (Variable.get(0).dependsOn(Ii)) {
                                sum += Variable.get(0).activate(input, Ii);
                                nothingDone = false;
                            }
                        }
                        if (nothingDone) {
                            return Variable.get(0).activate(input);
                        }
                        return sum;
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input) {
                        double sum = 0;
                        boolean nothingDone = true;
                        for (int Ii = 0; Ii < input.length; Ii++) {
                            if (Variable.get(0).dependsOn(Ii)) {
                                sum += Variable.get(0).activate(input, Ii);
                                nothingDone = false;
                            }
                        }
                        if (nothingDone) {
                            return Variable.get(0).activate(input);
                        }
                        return sum;
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index, final int j) {
                        return Variable.get(0).derive(input, index, j);
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index) {
                        return Variable.get(0).derive(input, index);
                    }
                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                }
                ,
                //8: prod:
                new NFOperation() {
                    @Override
                    public String expression() {
                        return "prod";
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input, int j) {
                        double prod = 1;
                        boolean nothingDone = true;
                        for (int Ii = 0; Ii < input.length; Ii++) {
                            if (Variable.get(0).dependsOn(Ii)) {
                                prod *= Variable.get(0).activate(input, Ii);
                                nothingDone = false;
                            }
                        }
                        if (nothingDone) {
                            return Variable.get(0).activate(input, j);
                        }
                        return prod;
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input) {
                        double prod = 1;
                        boolean nothingDone = true;
                        for (int Ii = 0; Ii < input.length; Ii++) {
                            if (Variable.get(0).dependsOn(Ii)) {
                                prod *= Variable.get(0).activate(input, Ii);
                                nothingDone = false;
                            }
                        }
                        if (nothingDone) {
                            return Variable.get(0).activate(input);
                        }
                        return prod;
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index, final int j) {
                        double u, ud, v, vd;
                        u = Variable.get(0).activate(input, 0);
                        ud = Variable.get(0).derive(input, index, 0);
                        System.out.println(ud);
                        for (int ji = 1; ji < input.length; ji++) {
                            v = Variable.get(0).activate(input, ji);
                            vd = Variable.get(0).derive(input, index, ji);
                            ud = u * vd + v * ud;
                            u *= v;
                        }
                        return ud;
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index) {
                        double u, ud, v, vd;
                        u = Variable.get(0).activate(input, 0);
                        ud = Variable.get(0).derive(input, index, 0);
                        System.out.println(ud);
                        for (int j = 1; j < input.length; j++) {
                            v = Variable.get(0).activate(input, j);
                            vd = Variable.get(0).derive(input, index, j);
                            ud = u * vd + v * ud;
                            u *= v;
                        }
                        return ud;
                    }
                }
                ,
                //9: abs:
                new NFOperation() {
                    @Override
                    public String expression() {
                        return "abs";
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input, int j) {
                        return Math.abs(Variable.get(0).activate(input, j));
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input) {
                        return Math.abs(Variable.get(0).activate(input));
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index, final int j) {
                        if (Variable.get(0).activate(input) < 0) {
                            return Variable.get(0).derive(input, index, j) * -1;
                        } else {
                            return Variable.get(0).derive(input, index, j);
                        }
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index) {
                        if (Variable.get(0).activate(input) < 0) {
                            return Variable.get(0).derive(input, index) * -1;
                        } else {
                            return Variable.get(0).derive(input, index);
                        }
                    }
                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                }
                ,
                //10: sin:
                new NFOperation() {
                    @Override
                    public String expression() {
                        return "sin";
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input, int j) {
                        return Math.sin(Variable.get(0).activate(input, j));
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input) {
                        return Math.sin(Variable.get(0).activate(input));
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index, final int j) {
                        return Math.cos(Variable.get(0).activate(input, j))
                                * Variable.get(0).derive(input, index, j);
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index) {
                        return Math.cos(Variable.get(0).activate(input))
                                * Variable.get(0).derive(input, index);
                    }
                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                }
                ,
                //10: cos:
                new NFOperation() {
                    @Override
                    public String expression() {
                        return "cos";
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input, int j) {
                        return Math.cos(Variable.get(0).activate(input, j));
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input) {
                        return Math.cos(Variable.get(0).activate(input));
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index, final int j) {
                        return -Math.sin(Variable.get(0).activate(input, j))
                                * Variable.get(0).derive(input, index, j);
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index) {
                        return -Math.sin(Variable.get(0).activate(input))
                                * Variable.get(0).derive(input, index);
                    }
                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                }
                ,
                //12: ^:
                new NFOperation() {
                    @Override
                    public String expression() {
                        return "^";
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input, int j) {
                        double result = Variable.get(0).activate(input, j);
                        for (int Vi = 1; Vi < Variable.size(); Vi++) {
                            final double current = Variable.get(Vi).activate(input, j);
                            result = Math.pow(result, current);
                        }
                        return result;
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input) {
                        double result = Variable.get(0).activate(input);
                        for (int Vi = 1; Vi < Variable.size(); Vi++) {
                            final double current = Variable.get(Vi).activate(input);
                            result = Math.pow(result, current);
                        }
                        return result;
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index, final int j) {
                        // d/dx(f(x)^g(x))=
                        //	f(x)^g(x) * d/dx(g(x)) * ln(f(x))
                        //	+ f(x)^(g(x)-1) * g(x) * d/dx(f(x))
                        double fg, dg, lnf;
                        double f = Variable.get(0).activate(input, j);
                        double df = Variable.get(0).derive(input, index, j);
                        double g;

                        for (int i = 0; i < Variable.size() - 2; i++) {
                            g = Variable.get(i + 1).activate(input, j);
                            fg = f * g;
                            dg = Variable.get(i + 1).derive(input, index, j);
                            lnf = Math.log(f);

                            df = fg * dg * lnf + f * (g - 1) * g * df;

                            f = Math.pow(f, g);
                        }
                        return df;
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index) {
                        // d/dx(f(x)^g(x))=
                        //f(x)^g(x) * d/dx( g(x) ) * ln( f(x) )
                        //+ f(x)^( g(x)-1 ) * g(x) * d/dx( f(x) )
                        double fg, dg, lnf;
                        double f = Variable.get(0).activate(input);
                        double df = Variable.get(0).derive(input, index);
                        double g;

                        for (int i = 0; i < Variable.size() - 2; i++) {
                            g = Variable.get(i + 1).activate(input);
                            fg = f * g;
                            dg = Variable.get(i + 1).derive(input, index);
                            lnf = Math.log(f);
                            df = fg * dg * lnf + f * (g - 1) * g * df;
                            f = Math.pow(f, g);
                        }
                        return df;
                    }
                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                }
                ,
                //13: /:
                new NFOperation() {
                    @Override
                    public String expression() {
                        return "/";
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input, int j) {
                        double result = Variable.get(0).activate(input, j);
                        for (int Vi = 1; Vi < Variable.size(); Vi++) {
                            final double current = Variable.get(Vi).activate(input, j);
                            result /= current;
                        }
                        return result;
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input) {
                        double result = Variable.get(0).activate(input);
                        for (int Vi = 1; Vi < Variable.size(); Vi++) {
                            final double current = Variable.get(Vi).activate(input);
                            result /= current;
                        }
                        return result;
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index, final int j) {
                        double u, ud, v, vd;
                        u = Variable.get(0).activate(input, j);
                        ud = Variable.get(0).derive(input, index, j);

                        for (int i = 0; i < Variable.size() - 1; i++) {
                            v = Variable.get(i + 1).activate(input, j);
                            vd = Variable.get(i + 1).derive(input, index, j);
                            ud = (ud * v - u * vd) / Math.pow(v, 2);
                            u /= v;
                        }
                        return ud;
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index) {
                        double derivative = 0;
                        double tempVar = Variable.get(0).activate(input);
                        derivative = Variable.get(0).derive(input, index);

                        for (int i = 0; i < Variable.size() - 1; i++) {
                            double u, ud, v, vd;
                            v = Variable.get(i + 1).activate(input);
                            vd = Variable.get(i + 1).derive(input, index);
                            u = tempVar;
                            ud = derivative;
                            derivative = (ud * v - u * vd) / Math.pow(v, 2);
                            tempVar /= v;
                        }
                        return derivative;
                    }
                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                }
                ,
                //14: *:
                new NFOperation() {
                    @Override
                    public String expression() {
                        return "*";
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input, int j) {
                        double result = Variable.get(0).activate(input, j);
                        for (int Vi = 1; Vi < Variable.size(); Vi++) {
                            final double current = Variable.get(Vi).activate(input, j);
                            result *= current;
                        }
                        return result;
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input) {
                        double result = Variable.get(0).activate(input);
                        for (int Vi = 1; Vi < Variable.size(); Vi++) {
                            final double current = Variable.get(Vi).activate(input);
                            result *= current;
                        }
                        return result;
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index, final int j) {
                        double u, ud, v, vd;
                        u = Variable.get(0).activate(input, j);
                        ud = Variable.get(0).derive(input, index, j);

                        for (int ji = 1; ji < Variable.size(); ji++) {
                            v = Variable.get(ji).activate(input, j);
                            vd = Variable.get(ji).derive(input, index, j);
                            System.out.println("ud" + (u * vd + v * ud) + "=u" + u + "*vd" + vd + "+v" + v + "*ud" + ud);
                            ud = u * vd + v * ud;
                            u *= v;
                        }
                        System.out.println("* d: " + ud + "; j: " + j);
                        return ud;
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index) {
                        double u, ud, v, vd;
                        u = Variable.get(0).activate(input);
                        ud = Variable.get(0).derive(input, index);

                        for (int j = 1; j < Variable.size(); j++) {
                            v = Variable.get(j).activate(input);
                            vd = Variable.get(j).derive(input, index);

                            ud = u * vd + v * ud;
                            u *= v;
                        }
                        return ud;
                    }
                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                }
                ,
                //15: %:
                new NFOperation() {
                    @Override
                    public String expression() {
                        return "%";
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input, int j) {
                        double result = Variable.get(0).activate(input, j);
                        for (int Vi = 1; Vi < Variable.size(); Vi++) {
                            final double current = Variable.get(Vi).activate(input, j);
                            result %= current;
                        }
                        return result;
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input) {
                        double result = Variable.get(0).activate(input);
                        for (int Vi = 1; Vi < Variable.size(); Vi++) {
                            final double current = Variable.get(Vi).activate(input);
                            result %= current;
                        }
                        return result;
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index, final int j) {
                        return Variable.get(0).derive(input, index, j);// j ?
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index) {
                        return Variable.get(0).derive(input, index);
                    }
                }
                ,
                //16: -:
                new NFOperation() {
                    @Override
                    public String expression() {
                        return "-";
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input, int j) {
                        double result = Variable.get(0).activate(input, j);
                        for (int Vi = 1; Vi < Variable.size(); Vi++) {
                            final double current = Variable.get(Vi).activate(input, j);
                            result -= current;
                        }
                        return result;
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input) {
                        double result = Variable.get(0).activate(input);
                        for (int Vi = 1; Vi < Variable.size(); Vi++) {
                            final double current = Variable.get(Vi).activate(input);
                            result -= current;
                        }
                        return result;
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index, final int j) {
                        double d = 0;
                        for (int i = 0; i < Variable.size(); ++i) {
                            if (Variable.get(i).dependsOn(index)) {
                                if (i == 0) {
                                    d += Variable.get(i).derive(input, index, j);
                                } else {
                                    d -= Variable.get(i).derive(input, index, j);
                                }
                            }
                        }
                        return d;
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index) {
                        double derivative = 0;
                        for (int i = 0; i < Variable.size(); ++i) {
                            if (Variable.get(i).dependsOn(index)) {
                                if (i == 0) {
                                    derivative += Variable.get(i).derive(input, index);
                                } else {
                                    derivative -= Variable.get(i).derive(input, index);
                                }
                            }
                        }
                        return derivative;
                    }
                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                }
                ,
                //17: +:
                new NFOperation() {
                    @Override
                    public String expression() {
                        return "+";
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input, int j) {
                        double result = Variable.get(0).activate(input, j);
                        for (int Vi = 1; Vi < Variable.size(); Vi++) {
                            final double current = Variable.get(Vi).activate(input, j);
                            result += current;
                        }
                        return result;
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double activate(final double[] input) {
                        double result = Variable.get(0).activate(input);
                        for (int Vi = 1; Vi < Variable.size(); Vi++) {
                            final double current = Variable.get(Vi).activate(input);
                            result += current;
                        }
                        return result;
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index, final int j) {
                        double d = 0;
                        for (int i = 0; i < Variable.size(); ++i) {
                            if (Variable.get(i).dependsOn(index)) {
                                d += Variable.get(i).derive(input, index, j);
                            }
                        }
                        return d;
                    }

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    @Override
                    public double derive(final double[] input, final int index) {
                        double derivative = 0;
                        for (int i = 0; i < Variable.size(); ++i) {
                            if (Variable.get(i).dependsOn(index)) {
                                derivative += Variable.get(i).derive(input, index);
                            }
                        }
                        return derivative;
                    }
                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                }
        };
        //---------------------------------------------------------
        if (Operations != null && Operations.size() >= 1) {
            System.out.println("Operation 0 : " + Operations.get(0));
            for (int k = 0; k < NVFHead.OperationRegister.length; ++k) {
                if (NVFHead.OperationRegister[k].equals(Operations.get(0))) {
                    Opertaion = OperationCollection[k];
                }
            }
        }
        if (Components.size() == 1) {
            System.out.println("Only one component left -> no operations! -> testing for function:");
            System.out.println("parsing at " + 0 + " ...(possably a function!)");
            String possibleFunction = utility.parsedOperation(Components.get(0).toLowerCase(), 0);
            System.out.println("Function ?: " + possibleFunction);
            if (possibleFunction != null) {
                if (possibleFunction.length() > 1) {
                    for (int Oi = 0; Oi < OperationRegister.length; Oi++) {
                        if (OperationRegister[Oi].equals(possibleFunction)) {
                            Opertaion = OperationCollection[Oi];
                            NVFunction newCore = new NVFHead();
                            newCore = newCore.newBuild(utility.parsedComponent(Components.get(0), possibleFunction.length()));
                            Variable.add(newCore);
                            return this;
                        }
                    }
                }
            }
            System.out.println("Function: " + possibleFunction);
            //---
            System.out.println("1 comonent left: -> unpackAndCorrect(component)");
            String component = utility.unpackAndCorrect(Components.get(0));
            System.out.println("component: " + component);
            System.out.println("Checking if component is variable (value/input): ");
            if ((component.charAt(0) <= '9' && component.charAt(0) >= '0') || component.charAt(0) == '-' || component.charAt(0) == '+') {
                NVFunction newFunction = (NVFunction) new NVFValueLeave();
                newFunction = newFunction.newBuild(component);
                System.out.println("is Value leave! -> return newValueLeave.newBuilt(component)");
                return newFunction;
            }
            if (component.charAt(0) == 'i' || component.charAt(0) == 'I') {
                NVFunction newFunction = (NVFunction) new NFInputLeave();
                newFunction = newFunction.newBuild(component);
                System.out.println("Value leave! -> return newInputLeave.newBuilt(component)");
                return newFunction;
            }
            System.out.println("Component is not of type Leave! -> component = utility.cleanedHeadAndTail(component); ");
            //If the component did not trigger variable creation: =>Cleaning!
            component = utility.cleanedHeadAndTail(component);

            NVFunction newBuild = (NVFunction) new NVFHead();
            System.out.println("new build: NVFunction newBuild = (NVFunction)new NVFHead();");
            System.out.println("newBuild = newBuild.newBuild(component);");
            newBuild = newBuild.newBuild(component);
            System.out.println("-> return newBuild;");
            return newBuild;
        } else {
            final ListIterator<String> ComponentIterator2 = Components.listIterator();
            while (ComponentIterator2.hasNext()) {
                final String currentComponent2 = ComponentIterator2.next();
                System.out.println("this.Input.e_add(newCore2.newBuild(" + currentComponent2 + ")); Input.size(): " + this.Variable.size());
                NVFunction newCore2 = (NVFunction) new NVFHead();
                newCore2 = newCore2.newBuild(currentComponent2);//Dangerous recursion lives here!
                this.Variable.add(newCore2);
                if (newCore2 != null) {
                    System.out.println("newCore2 != null");
                }
            }
            this.Variable.trimToSize();
            if (this.Variable.size() == 1) {
                return this.Variable.get(0);
            }
            if (this.Variable.size() == 0) {
                return null;
            }
            ArrayList<NVFunction> newVariable = new ArrayList<NVFunction>();
            for (int Vi = 0; Vi < Variable.size(); Vi++) {
                if (Variable.get(Vi) != null) {
                    newVariable.add(Variable.get(Vi));
                }
            }
            Variable = newVariable;
            return (NVFunction) this;
        }
    }


    //Interpretation functions end.
    //======================================================================================
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public double activate(final double[] input, double[] bias) {
        double[] newInput = new double[input.length];
        //System.out.println(input.length + " - "+bias.length);
        if (bias != null) {
            for (int Ii = 0; Ii < input.length; Ii++) {
                newInput[Ii] = input[Ii] + bias[Ii];
            }
        } else {
            for (int Ii = 0; Ii < input.length; Ii++) {
                newInput[Ii] = input[Ii];
            }
        }
        return activate(newInput);
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public double activate(final double[] input, int j) {
        return Opertaion.activate(input, j);
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public double activate(final double[] input) {
        return Opertaion.activate(input);
    }

    @Override
    public double derive(final double[] input, final double[] bias, final int index) {
        double[] newInput = new double[input.length];
        if (bias != null) {
            for (int Ii = 0; Ii < input.length; Ii++) {
                newInput[Ii] = input[Ii] + bias[Ii];
            }
        } else {
            for (int Ii = 0; Ii < input.length; Ii++) {
                newInput[Ii] = input[Ii];
            }
        }
        return derive(newInput, index);
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public double derive(final double[] input, final int index, final int j) {
        return Opertaion.derive(input, index, j);
    }

    //=============================================================
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public double derive(final double[] input, final int index) {
        return Opertaion.derive(input, index);
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //=============================================================
    @Override
    public boolean dependsOn(final int index) {
        for (int i = 0; i < this.Variable.size(); ++i) {
            if (this.Variable.get(i).dependsOn(index)) {
                return true;
            }
        }
        return false;
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public String expression() {
        String reconstructed = "";
        if (Variable.size() == 1 && Opertaion.expression().length() > 1) {
            String expression = Variable.get(0).expression();
            if (expression.charAt(0) == '(' && expression.charAt(expression.length() - 1) == ')') {
                return Opertaion.expression() + expression;
            }
            return Opertaion.expression() + "(" + expression + ")";
        }
        for (int i = 0; i < Variable.size(); ++i) {
            if (Variable.get(i) != null) {
                reconstructed = String.valueOf(reconstructed) + Variable.get(i).expression();
            } else {
                reconstructed = String.valueOf(reconstructed) + "(null)";
            }
            if (i != Variable.size() - 1) {
                reconstructed = String.valueOf(reconstructed) + Opertaion.expression();
            }
        }
        return "(" + reconstructed + ")";
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //---------------------------------------------------------------

    //Activation stage 1: Function determination!
    //============================================================================================================================================================================================
    public double ActivationOf(double input, int functionID, boolean derive) {
        switch (functionID) {
            case 0:
                return getReLuOf(input, derive);
            case 1:
                return getSigmoidOf(input, derive);
            case 2:
                return getTanhOf(input, derive);
            case 3:
                return getQuadraticOf(input, derive);
            case 4:
                return getLigmoidOf(input, derive);
            case 5:
                return getLinearOf(input, derive);
            case 6:
                return getGaussianOf(input, derive);
            default:
                return input;
        }
    }

    //Activation stage 1: Activation type determination
    //============================================================================================================================================================================================
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    public double getReLuOf(double input, boolean derive) {
        if (!derive) {
            return getReLuOf(input);
        } else {
            return getReLuDerivativeOf(input);
        }
    }

    public double getReLuOf(double Input) {
        double Output;
        if (Input >= 0) {
            Output = (Input + shift) * inclination;
        } else {
            Output = (Input + shift) * secondaryInclination;
        }
        return Output;
    }

    public double getReLuDerivativeOf(double Input) {
        double Output;
        if (Input >= 0) {
            Output = inclination;
        } else {
            Output = secondaryInclination;
        }
        return Output;
    }

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    public double getSigmoidOf(double input, boolean derive) {
        if (!derive) {
            return getSigmoidOf(input);
        } else {
            return getSigmoidDerivativeOf(input);
        }
    }

    public double getSigmoidOf(double Input) {
        double Output = 1 / (1 + Math.pow(Math.E, (-(Input + shift) * inclination)));
        return Output;
    }

    public double getSigmoidDerivativeOf(double Input) {
        double Output = inclination * (Math.pow(Math.E, -(Input + shift) * inclination)) / (Math.pow((1 + Math.pow(Math.E, -(Input + shift) * inclination)), 2) + 2 * Math.pow(Math.E, -(Input + shift) * inclination));
        return Output;
    }

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    public double getTanhOf(double input, boolean derive) {
        if (!derive) {
            return getTanhOf(input);
        } else {
            return getTanhDerivativeOf(input);
        }
    }

    public double getTanhOf(double Input) {
        double Output = ((Input + shift) * inclination) / Math.pow((1 + Math.pow(((Input + shift) * inclination), 2)), 0.5);
        return Output;
    }

    public double getTanhDerivativeOf(double Input) {
        double Output = (1 - Math.pow(((Input + shift) / Math.pow((1 + Math.pow((Input + shift), 2)), 0.5)), 2)) * inclination;
        return Output;
    }

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    public double getQuadraticOf(double input, boolean derive) {
        if (!derive) {
            return getQuadraticOf(input);
        } else {
            return getQuadraticDerivativeOf(input);
        }
    }

    public double getQuadraticOf(double Input) {
        double Output = ((Input + shift) * (Input + shift) * inclination);
        return Output;
    }

    public double getQuadraticDerivativeOf(double Input) {
        double Output = 2 * Input * inclination + 2 * shift * inclination;
        return Output;
    }

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    public double getLigmoidOf(double input, boolean derive) {
        if (!derive) {
            return getLigmoidOf(input);
        } else {
            return getLigmoidDerivativeOf(input);
        }
    }

    public double getLigmoidOf(double Input) {
        double Output = (inclination * (Input + shift) + (Math.log(Math.pow(Math.E, -(Input + shift) * inclination) + 1) / Math.log(Math.E)));
        return Output;
    }

    public double getLigmoidDerivativeOf(double Input) {
        double Output = inclination / (1 + Math.pow(Math.E, ((-Input + shift) * inclination)));
        return Output;
    }

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    public double getLinearOf(double input, boolean derive) {
        if (!derive) {
            return getLinearOf(input);
        } else {
            return getLinearDerivativeOf(input);
        }
    }

    public double getLinearOf(double Input) {
        double Output = inclination * (Input + shift);
        return Output;
    }

    public double getLinearDerivativeOf(double Input) {
        double Output = inclination;
        return Output;
    }

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    public double getGaussianOf(double input, boolean derive) {
        if (!derive) {
            return getGaussianOf(input);
        } else {
            return getGaussianDerivativeOf(input);
        }
    }

    public double getGaussianOf(double Input) {
        double Output = Math.pow(Math.E, -Math.pow(inclination * (Input + shift), 2));
        return Output;
    }

    public double getGaussianDerivativeOf(double Input) {
        double Output = -2 * (inclination * (Input + shift)) * Math.pow(Math.E, -Math.pow(inclination * (Input + shift), 2));
        return Output;
    }
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    private static class utility {

        public static int amountOfOperationsWithin(final List<String> operations) {
            int Count = 0;
            for (int i = 0; i < NVFHead.OperationRegister.length; ++i) {
                if (utility.containsOperation(NVFHead.OperationRegister[i], operations)) {
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
            for (int i = 0; i < NVFHead.OperationRegister.length; ++i) {
                System.out.print(OperationRegister[i] + " =?= " + operation + " -:|:- ");
                if (NVFHead.OperationRegister[i].equals(operation)) {
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

            equation = equation.replace("lig", OperationRegister[4]);
            equation = equation.replace("ligmoid", OperationRegister[4]);
            equation = equation.replace("softplus", OperationRegister[4]);
            equation = equation.replace("spls", OperationRegister[4]);
            equation = equation.replace("ligm", OperationRegister[4]);
            equation = equation.replace("linear", OperationRegister[5]);
            equation = equation.replace("sigmoid", OperationRegister[1]);
            equation = equation.replace("quadratic", OperationRegister[3]);
            equation = equation.replace("quadr", OperationRegister[3]);
            equation = equation.replace("gaussian", OperationRegister[6]);
            equation = equation.replace("gauss", OperationRegister[6]);
            equation = equation.replace("summation", OperationRegister[7]);
            equation = equation.replace("product", OperationRegister[8]);
            equation = equation.replace("absolute", OperationRegister[9]);
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
