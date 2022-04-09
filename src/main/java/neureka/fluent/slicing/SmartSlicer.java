package neureka.fluent.slicing;

import neureka.Tsr;
import neureka.framing.NDFrame;
import neureka.fluent.slicing.states.AxisOrGet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 *  This class is responsible for receiving any input and trying to interpret it so that a
 *  slice can be formed.
 */
public class SmartSlicer {

    /**
     *  An interface provided by sl4j which enables a modular logging backend!
     */
    private static final Logger _LOG = LoggerFactory.getLogger(SmartSlicer.class); // Why is this not final ? : For unit testing!

    public static <ValType> Tsr<ValType> slice(
            Object[] ranges,
            Tsr<ValType> source,
            SliceBuilder.CreationCallback<ValType> callback
    ) {
        AxisOrGet<ValType> sliceBuilder = new SliceBuilder<>(source, callback);
        List<Object> rangeList = new ArrayList<>();
        List<Integer> stepsList = new ArrayList<>();
        for (Object range : ranges ) {
            if ( range instanceof Map) {
                rangeList.addAll(((Map<?, ?>) range).keySet());
                stepsList.addAll(((Map<?, Integer>) range).values());
            }
            else if ( range instanceof int[] ) {
                List<Integer> intList = new ArrayList<>(((int[]) range).length);
                for ( int ii : (int[]) range ) intList.add(ii);
                rangeList.add(intList);
                stepsList.add(1);
            }
            else if ( range instanceof String[] ) {
                List<String> strList = new ArrayList<>(((String[]) range).length);
                strList.addAll(Arrays.asList((String[]) range));
                rangeList.add(strList);
                stepsList.add(1);
            }
            else if ( Iterable.class.isAssignableFrom( range.getClass() ) ) {
                Iterable<Object> iterableRange = (Iterable<Object>)  range;
                Iterator<Object> iterator = iterableRange.iterator();
                Object first = iterator.next();
                Object last = null;
                while (iterator.hasNext() ) {
                    last = iterator.next();
                }
                if ( last == null ) last = first;
                if ( first instanceof Number ) first = ((Number) first).intValue();
                if ( last instanceof Number ) last = ((Number) last).intValue();
                rangeList.add(Stream.of( first, last ).collect(Collectors.toList()));
                stepsList.add(1);
            }
            else {
                rangeList.add( range );
                stepsList.add(1);
            }
        }

        ranges = rangeList.toArray();

        for ( int i = 0; i < ranges.length; i++ ) {
            int first = 0;
            int last = 0;
            if ( !( ranges[ i ] instanceof  List ) ) {
                if ( ranges[ i ] instanceof Integer ) {
                    first = (Integer) ranges[ i ];
                    last = (Integer) ranges[ i ];
                } else {
                    NDFrame<?> frame = source.get( NDFrame.class );
                    if ( frame != null ) {
                        int position = frame.atAxis( i ).getIndexAtAlias( ranges[i] );
                                    //frame.get( ranges[ i ], i );
                        first = position;
                        last = position;
                    } else {
                        String message = "Given "+ NDFrame.class.getSimpleName()+" key at axis " + ( i ) + " not found!";
                        _LOG.error( message );
                        throw new IllegalStateException( message );
                    }
                }
            } else {
                ranges[ i ] = ( (List<?>) ranges[ i ] ).toArray();
                ranges[ i ] = ( ( (Object[]) ranges[ i ] )[ 0 ] instanceof List )
                        ? ( (List<?>) ( (Object[]) ranges[ i ] )[ 0 ] ).toArray()
                        : ( (Object[]) ranges[ i ] );
                if (
                        !( ( (Object[]) ( ranges[ i ] ) )[ 0 ] instanceof Integer )
                                || !( ( (Object[]) ranges[ i ] )[ ( (Object[]) ( ranges[ i ] ) ).length - 1 ] instanceof Integer )
                ) {
                    NDFrame<?> frame = source.get( NDFrame.class );
                    if ( !( ( (Object[]) (ranges[ i ]) )[ 0 ] instanceof Integer ) ) {
                        if ( frame != null ) {
                            first =
                                    frame
                                        .atAxis( i )
                                        .getIndexAtAlias( ( (Object[]) ranges[ i ])[ 0 ] );
                        }
                    }
                    else first = (Integer) ( (Object[]) ranges[ i ] )[ 0 ];

                    if ( !( ( (Object[]) ranges[ i ] )[ ( (Object[]) ranges[ i ] ).length - 1 ] instanceof Integer )  ) {
                        if ( frame != null ) {
                            last =
                                    frame
                                        .atAxis( i )
                                        .getIndexAtAlias(
                                                ( (Object[]) ranges[ i ] )[ ( (Object[]) ranges[ i ] ).length - 1 ]
                                        );
                        }
                    }
                    else last = (Integer) ( (Object[]) ranges[ i ] )[ ( (Object[]) ranges[ i ] ).length - 1 ];

                } else {
                    first = (Integer)( (Object[]) ranges[ i ] )[ 0 ];
                    last = (Integer) ( (Object[]) ranges[ i ] )[ ( (Object[]) ranges[ i ] ).length - 1 ];
                }
            }

            sliceBuilder =
                    sliceBuilder
                        .axis( i )
                        .from( first )
                        .to( last )
                        .step( stepsList.isEmpty() ? 1 : stepsList.get( i ) );

        }
        return sliceBuilder.get();
    }

}
