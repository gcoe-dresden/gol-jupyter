#pragma once

// return random colors
template <typename varTyp> struct Color_map {
  virtual int r(varTyp value) { return (2500 + 500 * value) % 65535; }
  virtual int g(varTyp value) { return (10408 + 500 * value) % 65535; }
  virtual int b(varTyp value) { return (7401 + 500 * value) % 65535; }
};

template <typename varTyp> struct Black_white_map : Color_map<varTyp> {
  int r(varTyp value) { return value ? 65535 : 0; }
  int g(varTyp value) { return value ? 65535 : 0; }
  int b(varTyp value) { return value ? 65535 : 0; }
};

// return 4 colors:
// white (11) == alive
// black ( 0) == dead
// green (10) == should be alive
// red   ( 1) == should be dead
// function to determine the state of a cell:  solution[i] * 10 + result[i]
// -> a dead cell is stored with 0 and a living cell with 1
struct Ghost_diff_map : Color_map<int> {
  int r(int value) {
    switch (value) {
    case 0:
      return 0;
    case 11:
      return 65535;
    case 1:
      return 65535;
    case 10:
      return 0;
    default:
      return 20000;
    }
  }

  int g(int value) {
    switch (value) {
    case 0:
      return 0;
    case 11:
      return 65535;
    case 1:
      return 0;
    case 10:
      return 65535;
    default:
      return 20000;
    }
  }

  int b(int value) {
    switch (value) {
    case 0:
      return 0;
    case 11:
      return 65535;
    case 1:
      return 0;
    case 10:
      return 0;
    default:
      return 20000;
    }
  }
};
