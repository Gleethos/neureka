package ct.autograd

import neureka.Neureka
import neureka.Tsr
import spock.lang.Specification
import testutility.UnitTester_Tensor

/**
 * These tests were originally Java test cases.
 * They have been ported to Spock tests almost without modification.
 */
class Autograd_Tensor_Component_Tests extends Specification
{

    def 'Test basic autograd behaviour. (Not on device)'()
    {
        given :
        Neureka.instance().reset();
        Neureka.instance().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(false);
        Neureka.instance().settings().view().setIsUsingLegacyView(true);

        Tsr x = new Tsr(new int[]{1}, 3).setRqsGradient(true);
        Tsr b = new Tsr(new int[]{1}, -4);
        Tsr w = new Tsr(new int[]{1}, 2);
        /**
         *      ((3-4)*2)^2 = 4
         *  dx:   8*3 - 32  = -8
         * */
        when : Tsr y = new Tsr(new Tsr[]{x, b, w}, "((i0+i1)*i2)^2");
        then : y.toString().contains("[1]:(4.0); ->d[1]:(-8.0), ");
        when : y.backward(new Tsr(2));
        then : x.toString().contains("-16.0");

        when : y = new Tsr("(","(",x,"+",b,")","*",w,")^2");
        then : y.toString().contains("[1]:(4.0); ->d[1]:(-8.0), ");
        when : y.backward(new Tsr(1));
        then : x.toString().contains("-24.0");

        when : y = new Tsr("((",x,"+",b,")*",w,")^2");
        then : y.toString().contains("[1]:(4.0); ->d[1]:(-8.0), ");
        when : y.backward(new Tsr(-1));
        then : x.toString().contains("-16.0");
        //===========================================
        when :
        Neureka.instance().settings().indexing().setIsUsingLegacyIndexing(true);
        x = new Tsr(
                new int[]{2, 3, 1},
                new double[]{
                        3, 2,
                        -1, -2,
                        2, 4
                }
        );
        y = new Tsr(
                new int[]{1, 3, 2},
                new double[]{
                        4, -1, 3,
                        2, 3, -1
                });
        Tsr z = new Tsr(new Tsr[]{x, y}, "I0xi1");
        then : z.toString().contains("[2x1x2]:(19.0, 22.0, 1.0, -6.0)");

        when : z = new Tsr(new Object[]{x, "x", y});
        then : z.toString().contains("[2x1x2]:(19.0, 22.0, 1.0, -6.0)");

        when :
        Neureka.instance().settings().indexing().setIsUsingLegacyIndexing(false);
        //Same test again but this time with reversed indexing:
        x = new Tsr(
                new int[]{2, 3, 1},
                new double[]{
                        3, 2, -1,
                        -2, 2, 4
                }
        );
        y = new Tsr(
                new int[]{1, 3, 2},
                new double[]{
                        4, -1,
                        3,  2,
                        3, -1
                });
        /*
                15, 2,
                10, 2
         */
        z = new Tsr(new Tsr[]{x, y}, "I0xi1");
        then : z.toString().contains("[2x1x2]:(15.0, 2.0, 10.0, 2.0)");
        when : z = new Tsr(new Object[]{x, "x", y});
        then : z.toString().contains("[2x1x2]:(15.0, 2.0, 10.0, 2.0)");
        //=======================
        when :
        Neureka.instance().settings().indexing().setIsUsingLegacyIndexing(true);
        x = new Tsr(
                new int[]{3, 3},
                new double[]{
                        1, 2, 5,
                        -1, 4, -2,
                        -2, 3, 4,
                }
        );
        y = new Tsr(
                new int[]{2, 2},
                new double[]{
                        -1, 3,
                        2, 3,
                }).setRqsGradient(true);

        then : y.toString().contains(":g:(null)");
        when : z = new Tsr(new Tsr[]{x, y}, "I0xi1");
        then : z.toString().contains("[2x2]:(15.0, 15.0, 18.0, 8.0)");

        when : z = new Tsr(new Object[]{x, "x", y});
        then : z.toString().contains("[2x2]:(15.0, 15.0, 18.0, 8.0)");

        when : z.backward(new Tsr(new int[]{2, 2}, 1));
        then : y.toString().contains("[2x2]:(-1.0, 3.0, 2.0, 3.0):g:(6.0, 9.0, 4.0, 9.0)");
        //---
        when :
        Neureka.instance().settings().indexing().setIsUsingLegacyIndexing(false);
        //--- again but now reverse: (outcome should not change...)
        x = new Tsr(
                new int[]{3, 3},
                new double[]{
                        1, 2, 5,
                        -1, 4, -2,
                        -2, 3, 4,
                }
        );
        y = new Tsr(
                new int[]{2, 2},
                new double[]{
                        -1, 3,
                        2, 3,
                }).setRqsGradient(true);

        then : y.toString().contains(":g:(null)");
        when : z = new Tsr(new Tsr[]{x, y}, "I0xi1");
        then : z.toString().contains("[2x2]:(15.0, 15.0, 18.0, 8.0)");

        when : z = new Tsr(new Object[]{x, "x", y});
        then : z.toString().contains("[2x2]:(15.0, 15.0, 18.0, 8.0)");

        when : z.backward(new Tsr(new int[]{2, 2}, 1));
        then : y.toString().contains("[2x2]:(-1.0, 3.0, 2.0, 3.0):g:(6.0, 9.0, 4.0, 9.0)");
        //====
        when :
        x = new Tsr(new int[]{1}, 3);
        b = new Tsr(new int[]{1}, -5);
        w = new Tsr(new int[]{1}, -2);
        z = new Tsr(new Tsr[]{x, b, w}, "I0*i1*i2");
        then : z.toString().contains("[1]:(30.0)");

        when :
        x = new Tsr(new int[]{1}, 4).setRqsGradient(true);
        b = new Tsr(new int[]{1}, 0.5);
        w = new Tsr(new int[]{1}, 0.5);
        y = new Tsr(new Tsr[]{x, b, w}, "(2^i0^i1^i2^2");
        then :
            y.toString().contains("[1]:(4.0);");
            y.toString().contains(" ->d[1]:(1.38629E0), ");
        //===
        //TODO: add tests using more then 1 function and check if the graph is build correctly!
    }


}
