//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.aruco;



// C++: class EstimateParameters
/**
 * Pose estimation parameters
 *
 * pattern Defines center this system and axes direction (default PatternPositionType::ARUCO_CCW_CENTER).
 * useExtrinsicGuess Parameter used for SOLVEPNP_ITERATIVE. If true (1), the function uses the provided
 * rvec and tvec values as initial approximations of the rotation and translation vectors, respectively, and further
 * optimizes them (default false).
 * solvePnPMethod Method for solving a PnP problem: see REF: calib3d_solvePnP_flags (default SOLVEPNP_ITERATIVE).
 *
 * @deprecated Use Board::matchImagePoints and cv::solvePnP
 *
 * SEE: PatternPositionType, solvePnP()
 */
@Deprecated
public class EstimateParameters {

    protected final long nativeObj;
    protected EstimateParameters(long addr) { nativeObj = addr; }

    public long getNativeObjAddr() { return nativeObj; }

    // internal usage only
    public static EstimateParameters __fromPtr__(long addr) { return new EstimateParameters(addr); }

    //
    // C++:   cv::aruco::EstimateParameters::EstimateParameters()
    //

    public EstimateParameters() {
        nativeObj = EstimateParameters_0();
    }


    //
    // C++: aruco_PatternPositionType EstimateParameters::pattern
    //

    public int get_pattern() {
        return get_pattern_0(nativeObj);
    }


    //
    // C++: void EstimateParameters::pattern
    //

    public void set_pattern(int pattern) {
        set_pattern_0(nativeObj, pattern);
    }


    //
    // C++: bool EstimateParameters::useExtrinsicGuess
    //

    public boolean get_useExtrinsicGuess() {
        return get_useExtrinsicGuess_0(nativeObj);
    }


    //
    // C++: void EstimateParameters::useExtrinsicGuess
    //

    public void set_useExtrinsicGuess(boolean useExtrinsicGuess) {
        set_useExtrinsicGuess_0(nativeObj, useExtrinsicGuess);
    }


    //
    // C++: int EstimateParameters::solvePnPMethod
    //

    public int get_solvePnPMethod() {
        return get_solvePnPMethod_0(nativeObj);
    }


    //
    // C++: void EstimateParameters::solvePnPMethod
    //

    public void set_solvePnPMethod(int solvePnPMethod) {
        set_solvePnPMethod_0(nativeObj, solvePnPMethod);
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:   cv::aruco::EstimateParameters::EstimateParameters()
    private static native long EstimateParameters_0();

    // C++: aruco_PatternPositionType EstimateParameters::pattern
    private static native int get_pattern_0(long nativeObj);

    // C++: void EstimateParameters::pattern
    private static native void set_pattern_0(long nativeObj, int pattern);

    // C++: bool EstimateParameters::useExtrinsicGuess
    private static native boolean get_useExtrinsicGuess_0(long nativeObj);

    // C++: void EstimateParameters::useExtrinsicGuess
    private static native void set_useExtrinsicGuess_0(long nativeObj, boolean useExtrinsicGuess);

    // C++: int EstimateParameters::solvePnPMethod
    private static native int get_solvePnPMethod_0(long nativeObj);

    // C++: void EstimateParameters::solvePnPMethod
    private static native void set_solvePnPMethod_0(long nativeObj, int solvePnPMethod);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
