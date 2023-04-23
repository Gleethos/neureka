package neureka.ndim.config;

import java.util.*;
import java.util.function.Predicate;
import java.util.stream.IntStream;

public enum NDTrait
{
    COMPACT(NDTrait::_isCompact),
    SIMPLE(NDTrait::_isSimple),
    ROW_MAJOR(NDTrait::_isRM),
    COL_MAJOR(NDTrait::_isCM),
    CONTINUOUS_MATRIX(NDTrait::_isContinuousMatrix),
    OFFSET_MATRIX(NDTrait::_isOffsetMatrix),;

    private final Predicate<NDConfiguration> _predicate;

    NDTrait( Predicate<NDConfiguration> predicate ) { _predicate = Objects.requireNonNull(predicate); }


    static List<NDTrait> traitsOf( NDConfiguration ndc ) {
        List<NDTrait> traits = new ArrayList<>();
        for ( NDTrait trait : NDTrait.values() )
            if ( trait._predicate.test( ndc ) ) traits.add( trait );

        return Collections.unmodifiableList( traits );
    }

    private static int _rightSpreadPadding(NDConfiguration ndc ) {
        int numberOf0Spreads = 0;
        for ( int i = ndc.rank() - 1; i >= 0; i-- ) {
            if ( ndc.spread( i ) == 0 ) numberOf0Spreads++;
            else break;
        }
        return numberOf0Spreads;
    }

    private static boolean _isCompact( NDConfiguration ndc ) {
        return
            IntStream.range(0, ndc.rank()).allMatch(i -> ndc.spread(i) == 1 || ndc.spread(i) == 0 )
                    &&
            IntStream.range(0, ndc.rank()).allMatch(i -> ndc.offset(i) == 0);
    }

    private static boolean _isSimple( NDConfiguration ndc ) {
        int[] simpleTranslation = ndc.getLayout().newTranslationFor(ndc.shape());
        return Arrays.equals(ndc.translation(), simpleTranslation)
                    &&
                Arrays.equals(ndc.indicesMap(), simpleTranslation)
                    &&
                _isCompact(ndc);
    }

    /**
     *  What does it mean to be row major essentially? <br>
     *  Well all it really means is that the last dimension of a nd-array configuration
     *  has a stride of 1! <br>
     *
     * @param ndc The configuration to check.
     * @return Whether the configuration is row major.
     */
    private static boolean _isRM( NDConfiguration ndc ) {
        /*
            We simply need to check if the last real dimension has a stride and spread of 1.
            This tells us that at the last dimension we have all elements in
            a contiguous block of memory.
        */
        int realRank = ndc.rank() - _rightSpreadPadding( ndc );
        if ( realRank < 1 ) return false;
        boolean translation = ndc.translation( realRank - 1 ) == 1;
        boolean spread      = ndc.spread(      realRank - 1 ) == 1;
        return translation && spread;
    }

    /**
     *  What does it mean to be column major essentially? <br>
     *  Well all it really means is that the second last dimension of a nd-array configuration
     *  has a stride of 1! <br>
     *
     * @param ndc The configuration to check.
     * @return Whether the configuration is column major.
     */
    private static boolean _isCM( NDConfiguration ndc ) {
        /*
           We simply need to check if the second last real dimension has a stride and spread of 1.
           This tells us that at the second last dimension we have all elements in
           a contiguous block of memory.
        */
        int realRank = ndc.rank() - _rightSpreadPadding( ndc );
        if ( realRank < 1 ) return false;
        boolean translation = ndc.translation( realRank - 2 ) == 1;
        boolean spread      = ndc.spread(      realRank - 2 ) == 1;
        return translation && spread;
    }

    private static boolean _last2DimensionsAreNotPermuted( NDConfiguration ndc ) {
        int realRank = ndc.rank() - _rightSpreadPadding( ndc );
        if ( realRank < 2 ) return true;
        int translationCol = ndc.translation( realRank - 2 );
        int translationRow = ndc.translation( realRank - 1 );
        return translationCol == 1 || translationRow == 1;
    }

    private static boolean _isContinuousMatrix(NDConfiguration ndc ) {
        boolean isMatrix = _isMatrix( ndc );
        if ( !isMatrix ) return false;
        int realRank = ndc.rank() - _rightSpreadPadding( ndc );
        if ( _isRM(ndc) ) {
            // We need to check if the there is no row offset!
            return ndc.offset( realRank - 1 ) == 0;
        } else {
            // We need to check if the there is no column offset!
            return ndc.offset( realRank - 2 ) == 0;
        }
    }

    private static boolean _isOffsetMatrix(NDConfiguration ndc) {
        boolean isMatrix = _isMatrix( ndc );
        if ( !isMatrix ) return false;
        int realRank = ndc.rank() - _rightSpreadPadding( ndc );
        if ( _isRM(ndc) ) {
            // We need to check if the there is no row offset!
            return ndc.offset( realRank - 1 ) != 0;
        } else if ( _isCM(ndc) ) {
            // We need to check if the there is no column offset!
            return ndc.offset( realRank - 2 ) != 0;
        }
        return false;
    }

    private static boolean _isMatrix( NDConfiguration ndc ) {
        boolean last2DimsNotPermuted = _last2DimensionsAreNotPermuted( ndc );
        if ( !last2DimsNotPermuted ) return false;
        int realRank = ndc.rank() - _rightSpreadPadding( ndc );
        if ( realRank < 2 ) return false;
        // Now we need to make sure the strides of the last 2 dimensions are both 1.
        int spreadCol = ndc.spread( realRank - 2 );
        int spreadRow = ndc.spread( realRank - 1 );
        return spreadCol == 1 && spreadRow == 1;
    }
}
