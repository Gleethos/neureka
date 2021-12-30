/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.array.operation;

public final class FillMatchingDual {

    /**
     * 2013-10-22: Was set to 128 (based on calibration) but I saw a dip in relative performance (java matrix
     * benchmark) at size 200. So I changed it to 256.
     */
    public static int THRESHOLD = 256;

}
