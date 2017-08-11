"use strict";

class PositionTracker {
    constructor(x, y, z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    getXYZString() {
        return "Current Position " + this.x + ", " + this.y + ", " + this.z;
    }

    getQuadKey() {
        return toQuad(this.x, this.y, this.z);
    }

    getQuadKeyString() {
        return "Current Position " + this.getQuadKey();
    }
};

function toQuad(x, y, z) {
    var quadkey = '';
    for ( var i = z-1; i >= 0; --i) {
        var bitmask = 1 << i;
        var digit = 0;
        if ((x & bitmask) !== 0) {
            digit |= 1;}
        if ((y & bitmask) !== 0) {
            digit |= 2;}
        quadkey += digit;
    }
    return quadkey;
};