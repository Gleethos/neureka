import neureka.Tsr;
import neureka.calculus.Function;
import neureka.calculus.factory.assembly.FunctionBuilder;
import org.junit.Test;
import util.NTester_Function;

public class CalculusTests {


    @Test
    public void test_scalar_calculus(){

        NTester_Function tester = new NTester_Function("Calculus-Testing: Scalar calculus");

        Function f = FunctionBuilder.build("1/I[0]", false);
        tester.testScalarDerivative(f, new double[]{2}, 0, -0.25,  "testing scalar derivative");
        tester.testScalarActivation(f, new double[]{2}, 0.5, "testing scalar activation");

        f = FunctionBuilder.build("I[0]+1/I[0]", false);
        tester.testScalarActivation(f, new double[]{2}, 2.5, "testing scalar activation");
        tester.testScalarDerivative(f, new double[]{-1}, 0, 0.0,  "testing scalar derivative");
        tester.testScalarDerivative(f, new double[]{-3}, 0, 0.8888888888888888,  "testing scalar derivative");
        tester.testScalarDerivative(f, new double[]{0.2}, 0, -23.999999999999996,  "testing scalar derivative");

        f = FunctionBuilder.build("(I[0]+1/I[0])^-I[0]", false);
        tester.testScalarActivation(f, new double[]{1}, 0.5, "testing scalar activation");
        tester.testScalarDerivative(f, new double[]{0.2}, 0, -0.5217778675999797,  "testing scalar derivative");

        f = FunctionBuilder.build("(cos(I[0]*5)/5+I[0])*(1+sin(I[0])/2)", false);
        tester.testScalarActivation(f, new double[]{3}, 3.049021713079475, "testing scalar activation");
        tester.testScalarActivation(f, new double[]{2.5}, 3.507365283517986, "testing scalar activation");
        tester.testScalarActivation(f, new double[]{0}, 0.2, "testing scalar activation");

        assert f.derive(new double[]{0  }, 0)==1.1;
        assert f.derive(new double[]{0.5}, 0)==0.646867884000033;
        assert f.derive(new double[]{1.6}, 0)==-0.00697440343353687;
        assert f.derive(new double[]{-4 }, 0)==3.9174193383745917;
        tester.testScalarDerivative(f, new double[]{0  },0, 1.1,  "testing scalar derivative");
        tester.testScalarDerivative(f, new double[]{0.5},0, 0.646867884000033,  "testing scalar derivative");
        tester.testScalarDerivative(f, new double[]{1.6}, 0, -0.00697440343353687,  "testing scalar derivative");
        tester.testScalarDerivative(f, new double[]{-4 }, 0, 3.9174193383745917,  "testing scalar derivative");

        f = FunctionBuilder.build("sum((cos(I[j]*5)/5+I[j])*(1+sin(I[j])/2))", false);
        double[] inputs = {0.0, 0.5, 1.6, -4.0};
        double[] expected = {1.1, 0.646867884000033, -0.00697440343353687, 3.9174193383745917};
        for(int i=0; i<expected.length; i++){
            tester.testScalarDerivative(f, inputs, i, expected[i],  "testing scalar derivative");
        }
        tester.close();
    }



    @Test
    public void test_tensor_calculus()
    {

        NTester_Function tester = new NTester_Function("Calculus-Testing: Function parsing and tensor calculus");
        //EXPRESSION TESTING:
        tester.testExpression("ig0*(igj)xI[g1]", "((Ig[0]*Ig[j])xIg[1])", "");
        tester.testExpression("sum(ij)", "sum(I[j])", "");
        tester.testExpression("sum(1*(4-2/ij))", "sum(1.0*(4.0-(2.0/I[j])))", "");
        tester.testExpression("quadratic(ligmoid(Ij))", "quad(lig(I[j]))", "");
        tester.testExpression("softplus(I[3]^(3/i1)/sum(Ij^2)-23+I0/i1)", "lig((((I[3]^(3.0/I[1]))/sum(I[j]^2.0))-23.0)+(I[0]/I[1]))", "");
        tester.testExpression("1+3+5-23+I0*45/(345-651^I3-6)", "(1.0+3.0+(5.0-23.0)+(I[0]*(45.0/(345.0-(651.0^I[3])-6.0))))", "");
        tester.testExpression("sin(23*i1)-cos(i0^0.3)+tanh(23)", "((sin(23.0*I[1])-cos(I[0]^0.3))+tanh(23.0))", "");
        tester.testExpression("2*3/2-1", "((2.0*(3.0/2.0))-1.0)", "");
        tester.testExpression("3x5xI[4]xI[3]", "(((3.0x5.0)xI[4])xI[3])", "");
        tester.testExpression("[1,0, 5,3, 4]:(tanh(i0xi1))", "([1,0,5,3,4]:(tanh(I[0]xI[1])))", "");
        tester.testExpression("[0,2, 1,3, -1](sig(I0))", "([0,2,1,3,-1]:(sig(I[0])))", "");
        tester.testExpression("I[0]<-I[1]->I[2]", "((I[0]<-I[1])->I[2])", "");
        tester.testExpression("quadratic(I[0]) -> I[1] -> I[2]", "((quad(I[0])->I[1])->I[2])", "");
        tester.testExpression("((tanh(i0)", "tanh(I[0])", "");
        tester.testExpression("($$(gaus(i0*()", "gaus(I[0])", "");
        tester.testExpression("rrlu(i0)", "relu(I[0])", "");
        tester.testExpression("th(i0)*gzs(i0+I1)", "(tanh(I[0])*gaus(I[0]+I[1]))", "");

        tester.testExpression("th(i0)dgzs(i0+I1)", "(tanh(I[0])dgaus(I[0]+I[1]))", "");
        tester.testExpression("ijdgzs(i0+I1)", "(I[j]dgaus(I[0]+I[1]))", "");
        tester.testExpression("ijssum(i0+Ij)", "(I[j]ssum(I[0]+I[j]))", "");

        tester.testExpression("i[0] d>> i[1]", "(I[0]d"+((char)187)+"I[1])", "");

        //ACTIVATION TESTING:
        double[] input1 = {};
        tester.testActivation("6/2*(1+2)", input1, 9, "");
        input1 = new double[]{2, 3.2, 6};
        tester.testActivation("sum(Ij)", input1, 11.2, "");
        input1 = new double[]{0.5, 0.5, 100};
        tester.testActivation("prod(Ij)", input1, 25, "");
        input1 = new double[]{0.5, 0.5, 10};
        tester.testActivation("prod(prod(Ij))", input1, (2.5 * 2.5 * 2.5), "");
        input1 = new double[]{5, 4, 3, 12};//12/4-5+2+3
        tester.testActivation("I3/i[1]-I0+2+i2", input1, (3), "");
        input1 = new double[]{-4, -2, 6, -3, -8};//-3*-2/(-8--4-2)
        tester.testActivation("i3*i1/(i4-i0-2)-sig(0)+tanh(0)", input1, (-1.5), "");
        input1 = new double[]{2, 3, -2};//-3*-2/(-8--4-2)
        tester.testDeriviation("(i0*i1)*i2", input1, 0, (-6), "");
        input1 = new double[]{2, 3, -2};//-3*-2/(-8--4-2)
        tester.testDeriviation("lig(i0*i1)*i2", input1, 0, (-5.985164261060192), "");
        input1 = new double[]{2, 3, -2};//-3*-2/(-8--4-2)
        tester.testDeriviation("prod(ij)", input1, 1, (-4), "");

        Tsr[] tsrs = new Tsr[]{new Tsr(new int[]{2}, new double[]{1, 2}), new Tsr(new int[]{2},new double[]{3, -4})};
        Tsr expected = new Tsr(new int[]{2}, new double[]{0.9701425001453319, -0.8944271909999159});
        tester.testActivation("tanh(sum(Ij))", tsrs, expected, "");

        expected = new Tsr(new int[]{2}, new double[]{0.31326168751822286, 0.6931471805599453});
        tester.testActivation("lig(prod(Ij-2))", tsrs, expected, "");

        tsrs = new Tsr[]{new Tsr(new int[]{2, 4}, new double[]{10, 12, 16, 21, 33, 66, 222, 15})};
        expected = new Tsr(new int[]{1, 2, 2, 2}, new double[]{8.000335406372896, 10.000045398899216, 14.000000831528373, 19.000000005602796, 31.000000000000032, 64.0, 220.0, 13.000002260326852});
        tester.testActivation("lig([-1, 0, -2, -2](Ij-2))", tsrs, expected, "");

        tsrs = new Tsr[]{
                new Tsr(new int[]{2}, new double[]{-1, 3}),
                new Tsr(new int[]{2}, new double[]{7, -1}),
                new Tsr(new int[]{2}, new double[]{2, 2}),
        };
        expected = new Tsr(new int[]{2}, new double[]{-0.0018221023888012912, 0.2845552390654007});
        tester.testDerivative("lig(i0*i1)*i2", tsrs, 1, expected, "");
        //---
        expected = new Tsr(new int[]{2}, new double[]{(-1+7*7*7+2*2*2), (3*3*3+-1+2*2*2)});
        tester.testActivation("sum(ij^3)", tsrs, expected, "");

        expected = new Tsr(new int[]{2}, new double[]{3*Math.pow(7, 2), 3*Math.pow(-1, 2)});
        tester.testDerivative("sum(ij^3)", tsrs, 1, expected, "");
        //---
        expected = new Tsr(new int[]{2}, new double[]{(1+7*7+2*2), (3*3+1+2*2)});
        tester.testActivation("sum(ij*ij)", tsrs, expected, "");

        expected = new Tsr(new int[]{2}, new double[]{2*Math.pow(7, 1), 2*Math.pow(-1, 1)});
        tester.testDerivative("sum(ij*ij)", tsrs, 1, expected, "");
        //--
        expected = new Tsr(new int[]{2}, new double[]{(4), (2)});
        tester.testActivation("sum(ij/2)", tsrs, expected, "");

        expected = new Tsr(new int[]{2}, new double[]{1.0/2, 1.0/2});
        tester.testDerivative("sum(ij/2)", tsrs, 1, expected, "");
        //--
        expected = new Tsr(new int[]{2}, new double[]{(1+9+4), (5+1+4)});
        tester.testActivation("sum(ij+2)", tsrs, expected, "");

        expected = new Tsr(new int[]{2}, new double[]{1.0, 1.0});
        tester.testDerivative("sum(ij+2)", tsrs, 1, expected, "");
        //--
        expected = new Tsr(new int[]{2}, new double[]{(-3+5+0), (1+-3+0)});
        tester.testActivation("sum(ij-2)", tsrs, expected, "");

        expected = new Tsr(new int[]{2}, new double[]{1.0, 1.0});
        tester.testDerivative("sum(ij-2)", tsrs, 1, expected, "");
        //--
        expected = new Tsr(new int[]{2}, new double[]{(8)*3, (4)*3});
        tester.testActivation("sum(sum(ij))", tsrs, expected, "");

        expected = new Tsr(new int[]{2}, new double[]{3.0, 3.0});
        tester.testDerivative("sum(sum(ij))", tsrs, 1, expected, "");
        //--
        expected = new Tsr(new int[]{2}, new double[]{(-14)*3, (-6)*3});
        tester.testActivation("sum(prod(ij))", tsrs, expected, "");

        expected = new Tsr(new int[]{2}, new double[]{-2.0, 6.0});
        tester.testDerivative("(prod(ij))", tsrs, 1, expected, "");

        expected = new Tsr(new int[]{2}, new double[]{-2.0*3, 6.0*3});
        tester.testDerivative("sum(prod(ij))", tsrs, 1, expected, "");
        //--
        expected = new Tsr(new int[]{2}, new double[]{(-14)%3, -0.0});
        tester.testActivation("(prod(ij))%3", tsrs, expected, "");

        tester.close();
    }











}
