import React from "react";
import { Color } from "../../../lib/inputTypes";
import { enUS } from "date-fns/locale";
export type Locale = typeof enUS;
export type DateRangePickerValue = {
    from?: Date;
    to?: Date;
    selectValue?: string;
};
export interface DateRangePickerProps extends Omit<React.HTMLAttributes<HTMLDivElement>, "value" | "defaultValue"> {
    value?: DateRangePickerValue;
    defaultValue?: DateRangePickerValue;
    onValueChange?: (value: DateRangePickerValue) => void;
    enableSelect?: boolean;
    minDate?: Date;
    maxDate?: Date;
    placeholder?: string;
    selectPlaceholder?: string;
    disabled?: boolean;
    color?: Color;
    locale?: Locale;
    enableClear?: boolean;
    enableYearNavigation?: boolean;
    weekStartsOn?: 0 | 1 | 2 | 3 | 4 | 5 | 6;
    children?: React.ReactElement[] | React.ReactElement;
}
declare const DateRangePicker: React.ForwardRefExoticComponent<DateRangePickerProps & React.RefAttributes<HTMLDivElement>>;
export default DateRangePicker;
