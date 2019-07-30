package neureka.core.utility;

import java.util.ArrayList;
import java.util.Random;

public class DataHelper<T> {
    public T[] updateArray(T[] Array, int index, boolean remove, Class<T> tClass) {
        if (Array == null) {
            if (remove == true) {
                return Array;
            } else {
                Array = (T[]) java.lang.reflect.Array.newInstance(tClass, 1);//(Type[])new Object[1];
                return Array;
            }
        }
        if (Array.length < index) {
            if (remove == true) {
                return Array;
            }
        }
        T[] oldArray = Array;
        if (remove == false) {
            Array = (T[]) java.lang.reflect.Array.newInstance(tClass, oldArray.length + 1);//(T[])new Object[oldBias.length + 1];
        } else {
            if (Array.length == 1) {
                Array = null;
                return Array;
            }
            Array = (T[]) java.lang.reflect.Array.newInstance(tClass, oldArray.length - 1); //(T[])new Object[oldBias.length - 1];
        }
        for (int Ii = 0; Ii < Array.length; Ii++) {
            if (remove == false) {
                if (index > Ii) {
                    Array[Ii] = oldArray[Ii];
                }
                if (index < Ii) {
                    Array[Ii] = oldArray[Ii - 1];
                }
            } else {
                if (index > Ii) {
                    Array[Ii] = oldArray[Ii];
                }
                if (index == Ii) {
                    Array[Ii] = oldArray[Ii + 1];
                }
                if (index < Ii) {
                    Array[Ii] = oldArray[Ii + 1];
                }
            }
        }
        return Array;
    }

    //===================================================================
    public int[] updateArray(int[] Array, int index, boolean remove) {
        if (Array == null) {
            if (remove == true) {
                return Array;
            } else {
                if (index == 0) {
                    Array = new int[1];
                }
                return Array;
            }
        }
        if (Array.length < index) {
            return Array;
        }
        int[] oldArray = Array;
        if (remove == false) {
            Array = new int[oldArray.length + 1];
        } else {
            if (Array.length == 1) {
                Array = null;
                return Array;
            }
            Array = new int[oldArray.length - 1];
        }
        for (int Ii = 0; Ii < Array.length; Ii++) {
            if (remove == false) {
                if (index > Ii) {
                    Array[Ii] = oldArray[Ii];
                }
                if (index == Ii) {
                    Array[Ii] = 0;
                }
                if (index < Ii) {
                    Array[Ii] = oldArray[Ii - 1];
                }
            } else {
                if (index > Ii) {
                    Array[Ii] = oldArray[Ii];
                }
                if (index == Ii) {
                    Array[Ii] = oldArray[Ii + 1];
                }
                if (index < Ii) {
                    Array[Ii] = oldArray[Ii + 1];
                }
            }
        }
        return Array;
    }

    //===================================================================
    public double[] updateArray(double[] Array, int index, boolean remove) {
        if (Array == null) {
            if (remove == true) {
                return Array;
            } else {
                if (index == 0) {
                    Array = new double[1];
                }
                return Array;
            }
        }
        if (Array.length < index) {
            return Array;
        }
        double[] oldArray = Array;
        if (remove == false) {
            Array = new double[oldArray.length + 1];
        } else {
            if (Array.length == 1) {
                Array = null;
                return Array;
            }
            Array = new double[oldArray.length - 1];
        }

        for (int Ii = 0; Ii < Array.length; Ii++) {
            if (remove == false) {
                if (index > Ii) {
                    Array[Ii] = oldArray[Ii];
                }
                if (index == Ii) {
                    Array[Ii] = 0;
                }
                if (index < Ii) {
                    Array[Ii] = oldArray[Ii - 1];
                }
            } else {
                if (index > Ii) {
                    Array[Ii] = oldArray[Ii];
                }
                if (index == Ii) {
                    Array[Ii] = oldArray[Ii + 1];
                }
                if (index < Ii) {
                    Array[Ii] = oldArray[Ii + 1];
                }
            }
        }
        return Array;
    }

    //===================================================================
    public boolean[] updateArray(boolean[] Array, int index, boolean remove) {
        if (Array == null) {
            if (remove == true) {
                return Array;
            } else {
                if (index == 0) {
                    Array = new boolean[1];
                }
                return Array;
            }
        }
        if (Array.length < index) {
            return Array;
        }
        boolean[] oldArray = Array;
        if (remove == false) {
            Array = new boolean[oldArray.length + 1];
        } else {
            if (Array.length == 1) {
                Array = null;
                return Array;
            }
            Array = new boolean[oldArray.length - 1];
        }
        for (int Ii = 0; Ii < Array.length; Ii++) {
            if (remove == false) {
                if (index > Ii) {
                    Array[Ii] = oldArray[Ii];
                }
                if (index == Ii) {
                    Array[Ii] = false;
                }
                if (index < Ii) {
                    Array[Ii] = oldArray[Ii - 1];
                }
            } else {
                if (index > Ii) {
                    Array[Ii] = oldArray[Ii];
                }
                if (index == Ii) {
                    Array[Ii] = oldArray[Ii + 1];
                }
                if (index < Ii) {
                    Array[Ii] = oldArray[Ii + 1];
                }
            }
        }
        return Array;
    }

    //===================================================================
    public byte[] updateArray(byte[] Array, int index, boolean remove) {
        if (Array == null) {
            if (remove == true) {
                return Array;
            } else {
                if (index == 0) {
                    Array = new byte[1];
                }
                return Array;
            }
        }
        if (Array.length < index) {
            return Array;
        }
        byte[] oldArray = Array;
        if (remove == false) {
            Array = new byte[oldArray.length + 1];
        } else {
            if (Array.length == 1) {
                Array = null;
                return Array;
            }
            Array = new byte[oldArray.length - 1];
        }
        for (int Ii = 0; Ii < Array.length; Ii++) {
            if (remove == false) {
                if (index > Ii) {
                    Array[Ii] = oldArray[Ii];
                }
                if (index == Ii) {
                    Array[Ii] = 0;
                }
                if (index < Ii) {
                    Array[Ii] = oldArray[Ii - 1];
                }
            } else {
                if (index > Ii) {
                    Array[Ii] = oldArray[Ii];
                }
                if (index == Ii) {
                    Array[Ii] = oldArray[Ii + 1];
                }
                if (index < Ii) {
                    Array[Ii] = oldArray[Ii + 1];
                }
            }
        }
        return Array;
    }

    //===================================================================
    public interface Orderer {
        Object[] order(Object[] node);
    }

    public static Orderer withoutDuplicates =
            (Object[] node) -> {
                ArrayList<Object> NodeList = new ArrayList<Object>();
                for (int Ni = 0; Ni < node.length; Ni++) {
                    for (int Nii = Ni + 1; Nii < node.length; Nii++) {
                        if (node[Ni] != null && node[Nii] != null) {
                            if (node[Ni] == node[Nii]) {
                                node[Nii] = null;
                            }
                        }
                    }
                    if (node[Ni] != null) {
                        NodeList.add(node[Ni]);
                    }
                }
                Object[] array = new Object[NodeList.size()];
                for (int i = 0; i < NodeList.size(); i++) {
                    array[i] = NodeList.get(i);
                }
                return array;
            };

    public static double getDoubleOf(long seed) {
        double newNumber;
        double divisor = 0;
        divisor = (Math.pow(10, getIntegerOf(seed - 1, 2)) / Math.pow(10, getIntegerOf(seed + 1, 2)));
        final int[] prims = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61};
        int rank = (int) Math.abs(seed % prims.length);
        int min = 3;
        newNumber = 0;
        while (newNumber == 0 || min >= 0) {
            rank = Math.abs(rank % prims.length);
            newNumber /= 2;
            newNumber += ((getIntegerOf(prims[rank] + seed - 4, 21) - 10))
                    * (getIntegerOf(seed + 4 - prims[rank], 3) - 1);
            if (divisor < 10) {
                divisor *= 10;
            } else {
                divisor /= 10;
            }
            newNumber /= divisor;
            if (newNumber != 0) {
                min--;
            }
            rank++;
        }
        return newNumber;
    }

    public static int getIntegerOf(long seed, int m) {
        return getIntegerOf(seed, m, (int) (seed * Math.pow(61, 0.5)));
    }

    public static int getIntegerOf(long seed, int m, int rank) {
        int[] prims = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61};
        rank = Math.abs(rank % prims.length);
        if (seed == 0) {
            seed = 6731299924898493759L;
        }
        if ((prims[rank] * seed + m) % 2 == 0) {
            seed *= -1;
        }
        return (int) (4438.274645895732 * seed * Math.pow(prims[rank], 0.5) + seed) % m;
    }

    public static double randomDouble() {
        Random dice = new Random();
        double newNumber;
        double divisor = 0;
        divisor = (Math.pow(10, (dice.nextInt() % 2)) / Math.pow(10, (dice.nextInt() % 2)));
        int min = 3;
        newNumber = 0;
        while (newNumber == 0 || min >= 0) {
            newNumber /= 2;
            newNumber += ((dice.nextInt(21) - 10)) * (dice.nextInt(3) - 1);
            if (divisor < 10) {
                divisor *= 10;
            } else {
                divisor /= 10;
            }
            newNumber /= divisor;
            if (newNumber != 0) {
                min--;
            }
        }
        return newNumber;
    }
}
